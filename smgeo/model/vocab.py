
####################
### Imports
####################

## Standard Libary
import os
import sys
import gzip
import json
from collections import Counter
from multiprocessing import Pool

## External Libraries
from tqdm import tqdm
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import vstack, hstack, csr_matrix

## Local Modules
from ..util.helpers import (flatten,
                            chunks)


####################
### Class Definition
####################

class Vocabulary(object):

    """
    Vocabulary class
    """

    def __init__(self,
                 text=True,
                 subreddits=True,
                 time=True,
                 text_vocab_max=None,
                 subreddit_vocab_max=None,
                 min_text_freq=0,
                 max_text_freq=None,
                 min_subreddit_freq=0,
                 max_subreddit_freq=None,
                 max_toks=None,
                 max_docs=None,
                 binarize_counter=False,
                 jobs=8):
        """
        Vocabulary Learner

        Args:
            text (bool): Whether to include text-based features
            subreddits (bool): Whether to include subreddit-based features
            time (bool): Whether to include time-based features
            text_vocab_max (int or None): Maximum number of tokens to keep in text vocabulary.
            subreddit_vocab_max (int or None): Maximum number of subreddits to keep in subreddit vocabulary
            min_text_freq (int): Minimum frequency of occurence for text tokens to be kept in vocab.
            max_text_freq (int): Maximum frequency of occurence for text tokens to be kept in vocab.
            min_subreddit_freq (int): Minimum frequency of occurence for subreddit tokens to be kept in vocab
            max_subreddit_freq (int): Maximum frequency of occurence for subreddit tokens to be kept in vocab
            max_toks (int or None): Only consider the first N tokens from a single document.
                                    If None (default), will take all tokens.
            max_docs (int or None): Only consider the most recent N documents from a user. If
                                                  None (default), will take all documents.
            binarize_counter (bool): If True, multiple usages of an n-gram by a single user only count as 
                                     a single occurrence toward reaching the minimum/maximum usage thresholds.
            jobs (int): Number of cores to use
        """
        ## Vocabulary Parameters
        self._use_text = text
        self._use_subreddits = subreddits
        self._use_time = time
        self._text_vocab_max = text_vocab_max
        self._subreddit_vocab_max = subreddit_vocab_max
        self._min_text_freq = min_text_freq
        self._max_text_freq = max_text_freq
        self._min_subreddit_freq = min_subreddit_freq
        self._max_subreddit_freq = max_subreddit_freq
        self._max_docs = max_docs
        self._max_toks = max_toks
        self._binarize_counter = binarize_counter
        self._jobs = jobs
        ## Workspace
        self._class_state = "untrained"
        self.feature_to_idx = dict()
        self.feature_names = set()
        self.text2vec = None
        self.sub2vec = None
        self.time2vec = None

    def __repr__(self):
        """
        Generate a human-readable description of the class.

        Args:
            None
        
        Returns:
            desc (str): Prettified representation of the class
        """
        return "Vocabulary()"

    def _create_dict_vectorizer(self,
                                vocab):
        """
        """
        feature_to_idx = dict((n, i) for i, n in enumerate(vocab))
        _count2vec = DictVectorizer(separator=":")
        _count2vec.vocabulary_ = feature_to_idx.copy()
        rev_dict = dict((y, x) for x, y in feature_to_idx.items())
        _count2vec.feature_names_ = [rev_dict[i] for i in range(len(rev_dict))]
        return _count2vec

    def get_ordered_vocabulary(self):
        """
        Get the vocabulary ordered by it's determined index

        Args:
            None
        
        Returns:
            ordered_vocab (list): Vocabular terms ordered by it's learned index mapping
        """
        return self.feature_names

    def _select_n_recent_documents(self,
                                   user_data):
        """
        Select the N most recent documents in user data based on the 
        class initialization parameters. Assumes that the input
        data has already been sorted from newest to oldest.

        Args:
            user_data (list of dict): Processed user data dictionaries
        
        Returns:
            user_data (list of dict): N most recent documents
        """
        ## Downsample Documents
        if self._max_docs is not None:
            user_data = user_data[:min(len(user_data), self._max_docs)]
        return user_data
    
    def _select_first_n_tokens(self,
                               user_data):
        """

        """
        ## Downsample Tokens
        if self._max_toks is not None:
            user_data_filt = []
            for u in user_data:
                u["text"] = u["text"][:self._max_toks]
                user_data_filt.append(u)
            return user_data_filt
        return user_data

    def _load_user_data(self,
                       filename):
        """
        Load preprocessed user data from disk.

        Args:
            filename (str): Path to .tar.gz file
        
        Returns:
            user_data (list of dict): Preprocessed, filtered user
                                      data
        """
        ## Load the GZIPed file
        with gzip.open(filename) as the_file:
            user_data = json.load(the_file)
        ## Data Amount Filtering
        user_data = self._select_n_recent_documents(user_data)
        user_data = self._select_first_n_tokens(user_data)
        return user_data
    
    def _load_and_count(self,
                        filename):
        """
        Load preprocessed data and count the features.

        Args:
            filename (str): Path to a preprocessed text data file for a user.
        
        Returns:
            fn_counts (dict): Token counts in the user's data file.
        """
        ## Load Data
        fn_data = self._load_user_data(filename)
        ## Text
        if self._use_text:
            tokens = [i["text"] for i in fn_data]
            tokens = list(filter(lambda i: len(i) > 0, tokens))
        else:
            tokens = []
        ## Subreddits
        if self._use_subreddits:
            subreddits = [i["subreddit"] for i in fn_data]
        else:
            subreddits = []
        ## Time of Data
        if self._use_time:
            times = [i["created_utc"]["hour"] for i in fn_data]
        else:
            times = []
        ## Initialize Counters
        if self._class_state == "untrained" and self._binarize_counter:
            token_counts = Counter(list(set(flatten(tokens))))
            subreddit_counts = Counter(list(set(subreddits)))
        else:
            token_counts = Counter(flatten(tokens))
            subreddit_counts = Counter(subreddits)
        hour_counts = Counter(times)
        ## Concatenate
        all_counts = {
            "text":token_counts,
            "subreddit":subreddit_counts,
            "time":hour_counts
        }
        return all_counts
    
    def fit(self,
            filenames,
            chunksize=10):
        """
        Learn a vocabulary from preprocessed data files.

        Args:
            filenames (list): List of preprocessed data files containing
                              "text_tokenized" lists.
            chunksize (int): Number of filenames to process in a multiprocessing
                             chunk before combining into the main counter object.
        
        Returns:
            self
        """
        ## Assign Class State
        self._class_state = "untrained"
        ## Initialize Vocab Counters
        counts = {
                "text":Counter(),
                "subreddit":Counter(),
        }
        ## Chunk Files
        file_chunks = list(chunks(filenames, chunksize))
        ## Initialize Multiprocessing
        mp = Pool(processes=self._jobs)
        ## Count Tokens Across Files
        for fn_chunk in tqdm(file_chunks,
                             desc="Counting N-Grams",
                             total=len(file_chunks),
                             file=sys.stdout):
            mp_counts = list(mp.map(self._load_and_count, fn_chunk))
            for fn_counts in mp_counts:
                counts["text"] += fn_counts["text"]
                counts["subreddit"] += fn_counts["subreddit"]
        ## Close mp
        mp.close()
        ## Filtering (Frequency)
        counts["text"] = dict((x, y) for x, y in counts["text"].items() if y >= self._min_text_freq)
        if self._max_text_freq is not None:
            counts["text"] = dict((x, y) for x, y in counts["text"].items() if y <= self._max_text_freq)
        counts["subreddit"] = dict((x, y) for x, y in counts["subreddit"].items() if y >= self._min_subreddit_freq)
        if self._max_subreddit_freq is not None:
            counts["subreddit"] = dict((x,y) for x, y in counts["subreddit"].items() if y <= self._max_subreddit_freq)
        counts["text"] = Counter(counts["text"])
        counts["subreddit"] = Counter(counts["subreddit"])
        ## Filtering (Max Size)
        if self._text_vocab_max is not None:
            good_text = counts["text"].most_common(self._text_vocab_max)
            counts["text"] = dict((x,y) for x, y in good_text)
        if self._subreddit_vocab_max is not None:
            good_subs = counts["subreddit"].most_common(self._subreddit_vocab_max)
            counts["subreddit"] = dict((x,y) for x, y in good_subs)
        ## Construct Vectorizers, Feature Indices
        self.feature_names = []
        self._feature_inds = {"text":[], "subreddit":[], "time":[]}
        if self._use_text:
            self.text2vec = self._create_dict_vectorizer(sorted(counts["text"].keys()))
            self._feature_inds["text"] = list(range(len(self.text2vec.feature_names_)))
            self.feature_names.extend(self.text2vec.feature_names_)
        if self._use_subreddits:
            self.sub2vec = self._create_dict_vectorizer(sorted(counts["subreddit"].keys()))
            self._feature_inds["subreddit"] = list(range(len(self.feature_names),
                                                         len(self.feature_names) + len(self.sub2vec.feature_names_)))
            self.feature_names.extend(self.sub2vec.feature_names_)
        if self._use_time:
            self.time2vec = self._create_dict_vectorizer(list(f"HOUR={i}" for i in range(24)))
            self._feature_inds["time"] = list(range(len(self.feature_names),
                                                    len(self.feature_names) + len(self.time2vec.feature_names_)))
            self.feature_names.extend(self.time2vec.feature_names_)
        ## Feature Index
        self.feature_to_idx = dict((feat, i) for i, feat in enumerate(self.feature_names))
        ## Update Class State
        self._class_state = "trained"
        return self
    
    def _vectorize_file(self,
                        filename):
        """

        """
        ## Load and Count User Data
        user_data = self._load_and_count(filename)
        ## Vectorize
        X = self._vectorize_user_data(user_data)
        return filename, X
    
    def _vectorize_user_data(self,
                             user_data):
        """

        """
        X = []
        if self._use_text:
            X.append(self.text2vec.transform(user_data["text"]))
        if self._use_subreddits:
            X.append(self.sub2vec.transform(user_data["subreddit"]))
        if self._use_time:
            X.append(self.time2vec.transform(dict((f"HOUR={x}",y) for x, y in user_data["time"].items())))
        X = hstack(X)
        return X

    def transform(self,
                  filenames):
        """

        """
        ## Get Vectorized Forms of Data
        mp = Pool(self._jobs)
        res = list(tqdm(mp.imap_unordered(self._vectorize_file, filenames),
                        total=len(filenames),
                        desc="Vectorization",
                        file=sys.stdout))
        mp.close()
        ## Parse
        filenames = [r[0] for r in res]
        X = vstack([r[1] for r in res])
        X = X.tocsr()
        ## Return
        return filenames, X

