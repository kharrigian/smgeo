
################
### Imports
################

## Standard Libary
import sys

## External Libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix, hstack

## Local Modules
from ..util.tokenizer import STOPWORDS, CONTRACTIONS

################
### Globals
################

## Subreddit Stopset (Based on Subscriber Count)
POPULAR_SUBREDDITS = ["announcements",
            "funny",
            "askreddit",
            "gaming",
            "pics",
            "aww",
            "science",
            "worldnews",
            "music",
            "movies",
            "videos",
            "todayilearned",
            "news",
            "iama",
            "gifs",
            "showerthoughts",
            "earthporn",
            "askscience",
            "jokes",
            "food",
            "explainlikeimfive",
            "books",
            "blog",
            "lifeprotips",
            "art",
            "mildlyinteresting",
            "diy",
            "sports",
            "nottheonion",
            "space",
            "gadgets",
            "television",
            "documentaries",
            "photoshopbattles",
            "getmotivated",
            "listentothis",
            "upliftingnews",
            "tifu",
            "internetisbeautiful",
            "history",
            "philosophy",
            "futurology",
            "oldschoolcool",
            "dataisbeautiful",
            "writingprompts",
            "personalfinance",
            "nosleep",
            "creepy",
            "twoxchromosomes",
            "memes"
            ]

################
### Nonlocalness Class
################

class Nonlocalness(object):
    
    """
    Non-localness Calculation for Preprocessed Geolocation Inference Data
    """

    def __init__(self,
                 vocabulary,
                 alpha=1e-5,
                 beta=1,
                 min_support=10):
        """
        Args:
            vocabulary (Vocabulary object): Initialized vocabulary class
            alpha (float): Smoothing parameter for probability calculations
            beta (float [0, 1]): How much to weight non-localness vs. frequency
            min_support (int): Minimum frequency (user-based) for keeping in dataset
                               during transformation
        """
        self._vocabulary = vocabulary
        self._alpha = alpha
        self._beta = beta
        self._min_support = min_support
    
    def _get_stopset(self,
                     feature_type):
        """
        Initialize Stopset of Features

        Args:
            feature_type (str): Either "text" or "subreddit"
        
        Returns:
            stopset (list): List of possible stopset terms
        """
        stopset = []
        if feature_type == "text":
            for s in STOPWORDS:
                if s in CONTRACTIONS:
                    s = CONTRACTIONS[s].split()
                else:
                    s = [s]
                stopset.extend(s)
            stopset = [f"TOKEN={t}" for t in stopset]
        elif feature_type == "subreddit":
            for s in POPULAR_SUBREDDITS:
                stopset.append(f"SUBREDDIT={s.lower()}")
        else:
            raise ValueError("feature_type must be either text or subreddit")
        return stopset

    def fit(self,
            X,
            y,
            feature_type):
        """
        Compute non-localness scores for a particular feature class.

        Args:
            X (2d-array): Feature Matrix
            y (1d-array): Discrete Classes
            feature_type (str): Either "text" or "subreddit"

        Returns:
            nl_scores_df (pandas DataFrame): Non-localness score per feature, per class
            agg_nl_scores_df (pandas DataFrame): Non-localness score per feature, with ranking
        """
        ## Identify Appropriate Stopset
        stopset = self._get_stopset(feature_type)
        ## Subset of Data Indices We Care About
        feature_type_ind = self._vocabulary._feature_inds[feature_type]
        stopset_feature_ind = sorted([self._vocabulary.feature_to_idx[s] for s in stopset if s in self._vocabulary.feature_to_idx])
        class_inds = dict((c, np.where(np.array(y)==c)[0]) for c in set(y))
        ## Prepare Data
        X_binary = (X>0).astype(int)
        ## Stopset Weights
        stopset_weights = X_binary[:, stopset_feature_ind].sum(axis=0) / X_binary[:, stopset_feature_ind].sum()
        stopset_weights = np.array(stopset_weights)[0]
        ## Probability Matrix (Prob(class | feature))
        p_c_u = np.zeros((len(class_inds), X_binary.shape[1]))
        for i, c in enumerate(sorted(class_inds)):
            p_c_u[i] += np.array(X_binary[class_inds[c]].sum(axis=0))[0]
        p_c_u = (p_c_u + self._alpha) / (p_c_u + self._alpha).sum(axis=0)
        ## Nonlocalness Scores
        nl_scores = np.zeros_like(p_c_u)
        for stop_ind, stop_weight in tqdm(zip(stopset_feature_ind, stopset_weights),
                                          total=len(stopset_feature_ind),
                                          file=sys.stdout,
                                          desc=f"Stopset Features ({feature_type})"):
            sim_skl = p_c_u[:, [stop_ind]] * np.log(p_c_u[:, [stop_ind]] / p_c_u) + \
                      p_c_u * np.log(p_c_u /  p_c_u[:, [stop_ind]])
            nl_scores += stop_weight * sim_skl
        ## Subset of NL Scores for Feature Type
        nl_scores_df = pd.DataFrame(index=sorted(class_inds),
                                    data=nl_scores[:,feature_type_ind],
                                    columns=np.array(self._vocabulary.feature_names)[feature_type_ind])
        ## Aggregate Scores
        agg_nl_scores_df = nl_scores_df.sum(axis=0).reset_index()
        agg_nl_scores_df = agg_nl_scores_df.rename(columns={"index":"feature",0:"score"})
        agg_nl_scores_df["frequency"] = np.array(X_binary[:, feature_type_ind].sum(axis=0))[0]
        ## Create Aggregate Rank
        agg_nl_scores_df["score_rank"] = agg_nl_scores_df["score"].rank(method="min")
        agg_nl_scores_df["frequency_rank"] = agg_nl_scores_df["frequency"].rank(method="min")
        agg_nl_scores_df["weighted_rank"] = self._beta * agg_nl_scores_df["score_rank"] + \
                                            (1 - self._beta) *  agg_nl_scores_df["frequency_rank"]
        return nl_scores_df, agg_nl_scores_df
    
    def transform(self,
                  X,
                  feature_type,
                  agg_nl_scores_df,
                  k_top):
        """
        Extract the top features for a given feature type in the original bow-representation
        feature matrix. 

        Args:
            X (2d-feature matrix): Sparse (csr) matrix
            feature_type (str): One of "text" or "subreddit"
            agg_nl_scores_df (pandas DataFrame): Non-localness score per feature, with ranking
            k_top (int): Number of top features to Keep
        """
        ## Identify Subset of Features to Keep, Abiding by Minimum Support Parameter
        top_feats = []
        count = 0
        for feature_idx, feature_score in agg_nl_scores_df.sort_values("weighted_rank", ascending=False).iterrows():
            if feature_score["frequency"] >= self._min_support:
                top_feats.append(feature_idx)
                count += 1
            if count == k_top:
                break
        top_feats = sorted(top_feats)
        ## Splice the Feature Matrix
        non_feature_types = [i for i in ["text","subreddit","time"] if i != feature_type]
        top_feature_type_inds = [self._vocabulary._feature_inds[feature_type][i] for i in top_feats]
        X_T = []
        feature_names = []
        cur_start_index = 0
        for ft in ["text","subreddit","time"]:
            ## Add Features
            if ft == feature_type:
                X_T.append(X[:, top_feature_type_inds])
                ft_feature_names = [self._vocabulary.feature_names[i] for i in top_feature_type_inds]
            else:
                X_T.append(X[:, self._vocabulary._feature_inds[ft]])
                ft_feature_names = [self._vocabulary.feature_names[i] for i in self._vocabulary._feature_inds[ft]]
            ## Update The Vocabulary Object
            new_ft_index = list(range(cur_start_index, cur_start_index + X_T[-1].shape[1]))
            self._vocabulary._feature_inds[ft] = new_ft_index
            cur_start_index += X_T[-1].shape[1]
            feature_names.extend(ft_feature_names)
            ## Re-initialize Dict Vectorizer Features
            if ft == feature_type:
                ft2vec_name = {"text":"text2vec","subreddit":"sub2vec","time":"time2vec"}[ft]
                setattr(self._vocabulary, ft2vec_name, self._vocabulary._create_dict_vectorizer(ft_feature_names))
        ## Stack The Transformed Data
        X_T = hstack(X_T).tocsr()
        ## Update The Vocabulary Object
        self._vocabulary.feature_names = feature_names
        self._vocabulary.feature_to_idx = dict((f, i) for i, f in enumerate(feature_names))
        return X_T