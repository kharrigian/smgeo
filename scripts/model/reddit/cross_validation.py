
########################
### Globals
########################

## Place for Storing Cross Validation Results
RESULTS_DIR = "./data/results/reddit/cross_validation/"

## Cross Validation Parameters
K_FOLDS = 5
TEST_SIZE = .2
STRATIFIED = True
CV_RANDOM_STATE = 42
RUN_ON_TEST = False
CACHE_MODELS = False

########################
### Imports
########################

## Standard Library
import os
import sys
import json
import gzip
import argparse
from copy import deepcopy
from datetime import datetime

## External Libraries
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import (KFold,
                                     StratifiedKFold,
                                     train_test_split)

## Local
from smgeo.model.vocab import Vocabulary
from smgeo.model.feature_selection import Nonlocalness
from smgeo.model.inference import GeolocationInference
from smgeo.util.logging import initialize_logger

## Initialize Logger
LOGGER = initialize_logger()

########################
### Helpers
########################

def parse_command_line():
    """
    Identify the data configuration and model configuration
    by parsing the command line

    Args:
        None 
    
    Returns:
        args (Argparse object): Parsed command line arguments
    """
    ## Initialize Parser Object
    parser = argparse.ArgumentParser(description="Reddit Geolocation Inference Cross-Validation")
    ## Generic Arguments
    parser.add_argument("data_config_path",
                         help="Path to data configuration JSON file")
    parser.add_argument("model_config_path",
                        help="Path to model configuration JSON file")
    ## Parse Arguments
    args = parser.parse_args()
    ## Check Arguments
    if not os.path.exists(args.data_config_path):
        raise FileNotFoundError(f"Could not find data_config_path: {args.data_config_path}")
    if not os.path.exists(args.model_config_path):
        raise FileNotFoundError(f"Could not find model_config_path: {args.model_config_path}")
    return args

def load_settings():
    """
    Load repository settings

    Args:
        None
    
    Returns:
        settings_config (dict): Path to relevant directories (e.g. data, models)
    """
    settings_file =  os.path.dirname(os.path.abspath(__file__)) + \
                     "/../../../configurations/settings.json"
    if not os.path.exists(settings_file):
        raise FileNotFoundError(f"Could not find setting file in expected location: {settings_file}")
    with open(settings_file, "r") as the_file:
        settings_config = json.load(the_file)
    return settings_config

def load_data_config(args):
    """
    Load data processing configuration

    Args:
        args (Argparse object): Command-line arguments
    
    Returns:
        data_config (dict): Data processing configuration
    """
    with open(args.data_config_path, "r") as the_file:
        data_config = json.load(the_file)
    return data_config

def load_model_config(args):
    """
    Load model training configuration

    Args:
        args (Argparse object): Command-line arguments
    
    Returns:
        model_config (dict): Model training parameters
    """
    with open(args.model_config_path, "r") as the_file:
        model_config = json.load(the_file)
    return model_config

def create_cross_validation_directory(model_name):
    """
    Create a directory for storing cross-validation outputs

    Args:
        model_name (str): Name of the experiment to use for naming the directory
    
    Returns:
        output_dir (str): Path to cross-validation output directory
    """
    runtime = datetime.strftime(datetime.utcnow(), "%Y_%m_%d_%H_%M")
    output_dir = f"{RESULTS_DIR}{runtime}_{model_name}/"
    os.makedirs(output_dir)
    return output_dir

def cache_run_parameters(run_dir,
                         data_config,
                         model_config):
    """
    Save the parameters used for the cross-validation experiment

    Args:
        run_dir (str): Path to cross-validation output directory
        data_config (dict): Data processing parameters
        model_config (dict): Model training parameters
    
    Returns:
        None, saves parameters to disk
    """
    config = {
        "data":data_config,
        "model":model_config,
        "cross_validation":{
            "k_folds":K_FOLDS,
            "test_size":TEST_SIZE,
            "stratified":STRATIFIED,
            "random_state":CV_RANDOM_STATE,
            "run_on_test":RUN_ON_TEST,
            "cache_models":CACHE_MODELS
        }
    }
    with open(f"{run_dir}config.json", "w") as the_file:
        json.dump(config, the_file)

def filter_labels_by_resolution(labels,
                                min_resolution):
    """
    Isolate Users With Desired Label Resolution

    Args:
        labels (pandas DataFrame): Label Dataframe
        min_resolution (str): Minimum label resolution required
    
    Returns:
        labels (pandas DataFrame): Filtered Label Dataframe
    """
    resolutions = ["locality",
                   'administrative_area_level_3',
                   'administrative_area_level_2',
                   'administrative_area_level_1',
                   'country',
                   'continent']
    ## Drop Labels Without Any Nominal Resolution
    labels = labels.loc[~labels[resolutions].isnull().all(axis=1)]
    ## Drop Labels Without Country/Continent Labels
    labels = labels.loc[(~labels.country.isnull()) & (~labels.continent.isnull())]
    ## Get Resolution Label
    labels_res = labels[resolutions].isnull().idxmin(axis=1)
    acceptable_res = []
    for r in resolutions:
        acceptable_res.append(r)
        if r == min_resolution:
            break
    labels_filtered = labels.loc[labels_res.isin(set(acceptable_res))]
    labels_filtered = labels_filtered.reset_index(drop=True).copy()
    return labels_filtered

def load_metadata(settings,
                  author_list):
    """
    Load Author Metadata

    Args:
        author_list (list of str): Names of the reddit users

    Returns:
        metadata (pandas DataFrame): Metadata dataframe
    """
    ## Settings
    AUTHORS_PROCESSED_DIR = settings.get("reddit").get("AUTHORS_PROCESSED_DIR")
    ## Metadata Files
    meta_files = list(map(lambda i: f"{AUTHORS_PROCESSED_DIR}{i}.meta.json.gz", author_list))
    ## Load Meta
    metadata = []
    for m in tqdm(meta_files, file=sys.stdout, desc="Metadata Files"):
        with gzip.open(m, "r") as the_file:
            mdata = json.load(the_file)
        metadata.append(mdata)
    ## Format
    metadata = pd.DataFrame(metadata)
    return metadata

def filter_labels_by_comments(metadata,
                              labels,
                              min_comments):
    """
    Filter out labeled users who do not have enough comments
    according to our criteria for modeling

    Args:
        metadata (pandas DataFrame): Metadata dataframe
        labels (pandas DataFrame): Labels dataframe
        min_comments (int): Minimum number of comments in history
    
    Returns:
        labels (pandas DataFrame): Filtered label dataframe
    """
    ## Filter Metadata
    metadata = metadata.loc[metadata["num_comments"] >= min_comments]
    metadata = metadata.reset_index(drop=True).copy()
    ## Filter Labels
    labels = labels.loc[labels["author"].isin(metadata["author"])]
    labels = labels.reset_index(drop=True).copy()
    return labels

def aggregate_location_hierarchy(labels,
                                 min_support = 10,
                                 index_col="source",
                                 ordered_levels = ['locality',
                                                   'administrative_area_level_1',
                                                   'country',
                                                   'continent']
                                 ):
    """
    Aggregate locations up to meet minimum size across labeled 
    dataset

    Args:
        labels (pandas DataFrame): Label dataframe
        min_support (int): Minimum number of users at location level
        index_col (str): How to index the returned result
        ordered_levels (list): Location hierarchy levels in order
                               to consider
    
    Returns:
        labels_aggregated (pandas Series): Aggregated series
    """
    ## Create a Copy
    labels_copy = labels.set_index(index_col).copy()
    ## Replace Missing Values Temporarily
    for o in ordered_levels:
        labels_copy[o] = labels_copy[o].fillna("--")
    ## Create Aggregated Labels (First Pass)
    labels_aggregated = []
    for ol_high in ordered_levels[:-1][::-1]:
        ol_levels = []
        for ol_low in ordered_levels[::-1]:
            ol_levels.append(ol_low)
            if ol_low == ol_high:
                break
        agg_tuple = labels_copy[ol_levels].apply(tuple, axis=1)
        agg_counts = agg_tuple.value_counts()
        failing_labels = labels_copy.loc[agg_tuple.isin(agg_counts.loc[agg_counts < min_support].index)].copy()
        failing_labels["agg_label"] = failing_labels[ol_levels[:-1]].apply(tuple, axis=1)
        labels_aggregated.append(failing_labels)
        labels_copy = labels_copy.loc[~labels_copy.index.isin(failing_labels.index)].copy()
    labels_copy["agg_label"] = agg_tuple.loc[labels_copy.index]
    labels_aggregated.append(labels_copy)
    labels_aggregated = pd.concat(labels_aggregated)["agg_label"]
    ## Round Up Until Support Reached
    for _ in range(len(ordered_levels)):
        final_map = {}
        for lbl, count in labels_aggregated.value_counts().items():
            if count >= min_support:
                final_map[lbl] = lbl
            else:
                lbl_list = list(lbl)
                if len(lbl_list) > 1:
                    lbl_list = lbl_list[:-1]
                    lbl_list_copy = []
                    for l in lbl_list:
                        if l != "--":
                            lbl_list_copy.append(l)
                        else:
                            break
                    lbl_list = lbl_list_copy
                final_map[lbl] = tuple(lbl_list)
        labels_aggregated = labels_aggregated.map(lambda l: final_map[l])
        if labels_aggregated.value_counts().min() >= min_support:
            break
    labels_aggregated = labels_aggregated.map(lambda i: ", ".join(list(i)[::-1]))
    labels_aggregated = labels_aggregated.str.lstrip("--, ")
    return labels_aggregated

def update_vocabulary(vocabulary,
                      X,
                      use_text=True,
                      use_subreddit=True,
                      use_time=True):
    """
    Select a subset of features to use

    Args:
        vocabulary (Vocabulary): Original vocabulary object
        X (csr_matrix): Full sparse feature matrix
        use_text (bool): If True, maintain text features if they exist
        use_subreddit (bool): If True, maintain subreddit features if they exist
        use_time (bool): If True, maintain time features if they exist

    Returns:
        vocabulary (Vocabulary): Copy of vocabulary with updated indices, attributes
        X (csr_matrix): Spliced feature matrix with subset of desired features
    """
    ## Make a Copy of the Vocabulary Object
    vocabulary = deepcopy(vocabulary)
    ## Identify Feature Indices in X
    new_feature_names = []
    new_feature_ind = {"text":[],"subreddit":[],"time":[]}
    X_splice = []
    cur_index = 0
    for ftype, flag in zip(["text","subreddit","time"], [use_text, use_subreddit, use_time]):
        if not flag:
            setattr(vocabulary, f"_use_{ftype}", flag)
            continue
        ftype_features = [vocabulary.feature_names[i] for i in vocabulary._feature_inds[ftype]]
        X_splice.extend(vocabulary._feature_inds[ftype])
        new_feature_ind[ftype] = list(range(cur_index, cur_index + len(ftype_features)))
        new_feature_names.extend(ftype_features)
        cur_index += len(ftype_features)
    ## Update X
    X = X[:,X_splice]
    ## Update Vocabulary
    vocabulary._feature_inds = new_feature_ind
    vocabulary.feature_names = new_feature_names
    vocabulary.feature_to_idx = dict((f, i) for i, f in enumerate(new_feature_names))
    return vocabulary, X

def distance_between(x, y):
    """
    Calculate the geodesic distance in miles between two arrays of coordinates

    Args:
        x (2d-array): [lon, lat] coordinates
        y (2d-array): [lon, lat] coordinates

    Returns:
        dist (1d-array): Geodesic distance
    """
    dist = np.array(list(map(lambda i: geodesic(i[0], i[1]).miles, zip(x[:,::-1], y[:,::-1]))))
    return dist

def load_labels(settings,
                data_config):
    """
    Load automatically curated labels

    Args:
        settings (dict): Repository settings file
        data_config (dict): Data processing configuration
    
    Returns:
        labels (pandas DataFrame): Author location labels and history metadata
    """
    LOGGER.info("Loading Labels and Metadata")
    ## Directories from Settings
    LABELS_DIR = settings.get("reddit").get("LABELS_DIR")
    AUTHORS_PROCESSED_DIR = settings.get("reddit").get("AUTHORS_PROCESSED_DIR")
    ## Data Parameters
    MIN_RESOLUTION = data_config.get("MIN_RESOLUTION")
    MIN_COMMENTS = data_config.get("MIN_COMMENTS")
    ## Load Labels
    label_file = f"{LABELS_DIR}author_labels.json.gz"
    labels = pd.read_json(label_file)
    labels["source"] = labels["author"].map(lambda author: f"{AUTHORS_PROCESSED_DIR}{author}.comments.json.gz")
    ## Filter Data Set by Resolution
    labels = filter_labels_by_resolution(labels, MIN_RESOLUTION)
    ## Load Metadata
    metadata = load_metadata(settings, labels["author"].tolist())
    ## Filter Data Set by Number of Commnets
    labels = filter_labels_by_comments(metadata, labels, MIN_COMMENTS)
    ## Merge Number of Comments
    labels = pd.merge(labels, metadata[["author","num_comments"]], how = "left", on = "author")
    return labels

def prepare_feature_set(settings,
                        data_config,
                        labels):
    """
    Learn vocabulary and create training feature set

    Args:
        settings (dict): Repository settings (e.g. data directories)
        data_config (dict): Data processing paramters
        labels (pandas DataFrame): User location labels

    Returns:
        vocabulary (Vocabulary class): Learned vocabulary object
        X (csr matrix): Feature matrix
        files (list): List of filenames associated with each row in X
        VOCAB_PARAMETERS (dict): Vocabulary parameters
        MIN_RESOLUTION (str): Resolution used for filtering out users
        MIN_COMMENTS (int): Minimum comments required in user history not to filter
    """
    ## Settings
    DATA_CACHE_DIR = settings.get("reddit").get("DATA_CACHE_DIR")
    ## Data Configuration Parameters
    DATA_CACHE_NAME = data_config.get("NAME")
    VOCAB_PARAMETERS = data_config.get("VOCAB_PARAMETERS")
    MIN_RESOLUTION = data_config.get("MIN_RESOLUTION")
    MIN_COMMENTS = data_config.get("MIN_COMMENTS")
    ## Cache Directory
    data_cache_dir = f"{DATA_CACHE_DIR}{DATA_CACHE_NAME}/"
    ## Preprocess Data if Necessary, Otherwise Load Cached Version
    if not os.path.exists(data_cache_dir):
        LOGGER.info("Beginning Data Preprocessing")
        ## Create Directory
        os.makedirs(data_cache_dir)
        ## Identify Processed Data Files
        data_files = labels["source"].tolist()
        ## Learn Feature Vocabulary
        vocabulary = Vocabulary(**VOCAB_PARAMETERS)
        vocabulary = vocabulary.fit(data_files)
        ## Vectorize Training Data
        files, X = vocabulary.transform(data_files)
        ## Cache Data
        joblib.dump(vocabulary, f"{data_cache_dir}vocabulary.joblib", compress=3)
        joblib.dump(X, f"{data_cache_dir}X.joblib", compress=3)
        joblib.dump(files, f"{data_cache_dir}files.joblib", compress=3)
        VOCAB_PARAMETERS.update({"min_resolution":MIN_RESOLUTION,"min_comments":MIN_COMMENTS})
        joblib.dump(VOCAB_PARAMETERS, f"{data_cache_dir}vocab_parameters.joblib", compress=3)
    else:
        LOGGER.info("Loading Preprocessed Data")
        vocabulary = joblib.load(f"{data_cache_dir}vocabulary.joblib")
        X = joblib.load(f"{data_cache_dir}X.joblib")
        files = joblib.load(f"{data_cache_dir}files.joblib")
        VOCAB_PARAMETERS = joblib.load(f"{data_cache_dir}vocab_parameters.joblib")
        MIN_RESOLUTION = VOCAB_PARAMETERS.pop("min_resolution", None)
        MIN_COMMENTS = VOCAB_PARAMETERS.pop("min_comments", None)
    return vocabulary, X, files, VOCAB_PARAMETERS, MIN_RESOLUTION, MIN_COMMENTS

def filter_dataset_by_resolution(model_config,
                                 labels,
                                 files,
                                 X):
    """
    Isolate users whose location meets a certain resolution criteria (e.g. city, state)

    Args:
        model_config (dict): Model training parameters
        labels (pandas Dataframe): User location labels
        files (list): List of processed data files associated with rows in X
        X (csr_matrix): Filtered feature matrix
    
    Returns:
        labels (pandas DatAFrame): Filtered location labels
        files (list): List of filenames aligned with rows in X
        X (csr_matrix): Feature matrix (filtered)
    """
    LOGGER.info("Applying Location and Resolution Filters")
    ## Configuration
    FILTER_TO_US = model_config.get("FILTER_TO_US")
    MIN_MODEL_RESOLUTION = model_config.get("MIN_MODEL_RESOLUTION")
    ## Filter Data to US
    if FILTER_TO_US:
        labels_ind = labels.set_index("source").loc[files]
        US_mask = np.where(
                (labels_ind["country"] == "United States") & \
                (~labels_ind["administrative_area_level_1"].isin(["Hawaii","Alaska"])))[0]
        files = [files[i] for i in US_mask]
        X = X[US_mask]
    ## Resolution Filtering
    labels = filter_labels_by_resolution(labels, MIN_MODEL_RESOLUTION)
    good_files = set(labels["source"])
    resolution_mask = [i for i, f in enumerate(files) if f in good_files]
    files = [files[i] for i in resolution_mask]
    X = X[resolution_mask]
    return labels, files, X

def create_split_dict(model_config,
                      labels,
                      files,
                      output_dir):
    """
    Identify held-out test set and cross-validation splits

    Args:
        model_config (dict): Model training parameters
        labels (pandas DataFrame): User location labels
        files (list): User identifying files 
        output_dir (str): Path to cross validation output directory
                          for saving splits
    
    Returns:
        split_dict (dict): Mapping between train, dev, test splits
    """
    LOGGER.info("Splitting Dataset")
    ## Model Parameters
    FILTER_TO_US = model_config.get("FILTER_TO_US")
    ## Identify Held-out Test Set
    train_dev_files, test_files = train_test_split(files,
                                                test_size=TEST_SIZE,
                                                random_state=CV_RANDOM_STATE)

    ## Create Cross-Validation Splits Within Train/Dev Files
    if STRATIFIED:
        strat_level = "administrative_area_level_1" if FILTER_TO_US else "continent"
        splitter = StratifiedKFold(n_splits=K_FOLDS,
                                shuffle=True,
                                random_state=CV_RANDOM_STATE)
        splits = splitter.split(train_dev_files,
                                labels.set_index("source").loc[train_dev_files][strat_level].tolist())
    else:
        splitter = KFold(n_splits=K_FOLDS,
                        shuffle=True,
                        random_state=CV_RANDOM_STATE)
        splits = splitter.split(train_dev_files)
    ## Explicitly Invoke Splits
    split_dict = {"test":{0:test_files},
                "train":{},
                "dev":{}}
    for fold, (train_, dev_) in enumerate(splits):
        split_dict["train"][fold] = [train_dev_files[t] for t in train_]
        split_dict["dev"][fold] = [train_dev_files[d] for d in dev_]
    ## Cache Splits
    with open(f"{output_dir}splits.json", "w") as the_file:
        json.dump(split_dict, the_file)
    return split_dict

def run_training(model_config,
                 train_files,
                 test_files,
                 train_ind,
                 test_ind,
                 files,
                 labels,
                 X,
                 vocabulary):
    """
    Train model on a subset of users, evaluate on train and dev

    Args:
        model_config (dict): Model training parameters
        train_files (list): Files to use for training
        test_files (list): Files to use for development
        train_ind (list): Indices in X to use for training
        test_ind (list): Indices in X to use for development
        files (list): Full list of user files in X
        labels (pandas DataFrame): Raw user location annotations 
        X (csr_matrix): Feture matrix
        vocabulary (Vocabulary): Pre-fit Vocabulary
    
    Returns:
        text_nl_scores (pandas DataFrame): Raw non-localness scores for text
        text_agg_nl_scores (pandas DataFrame): Text non-localness scores aggregated over location
        sub_nl_scores (pandas DataFrame): Raw non-localness scores for subreddits
        sub_agg_nl_scores (pandas DataFrame): Subreddit non-localness scores agreegated over location
        geo (GeolocationInferenceModel): Trained geolocation inference model
        train_res (dict): Training predictions and ground truth
        dev_res (dict): Development predictions and ground truth
    """
    ## Model Parameters
    USE_TEXT = model_config.get("USE_TEXT")
    USE_SUBREDDIT = model_config.get("USE_SUBREDDIT")
    USE_TIME = model_config.get("USE_TIME")
    MIN_SUPPORT = model_config.get("MIN_SUPPORT")
    TOP_K_TEXT = model_config.get("TOP_K_TEXT")
    TOP_K_SUBREDDITS = model_config.get("TOP_K_SUBREDDITS")
    ## Get Labels
    LOGGER.info("Isolating Coordinate Labels")
    y_train = labels.set_index("source").loc[[files[i] for i in train_ind]][["longitude","latitude"]].values
    y_test = labels.set_index("source").loc[[files[d] for d in test_ind]][["longitude","latitude"]].values
    ## Get Discrete Labels (for Feature Selection)
    LOGGER.info("Discretizing Training Labels")
    labels_train_agg = aggregate_location_hierarchy(labels.loc[labels["source"].isin(train_files)],
                                                    min_support=10,
                                                    index_col="source",
                                                    ordered_levels=['administrative_area_level_1',
                                                                    'country',
                                                                    'continent'])
    y_train_discrete = labels_train_agg.loc[[files[i] for i in train_ind]].values
    ## Count Support
    n_comments_train = labels.set_index("source").loc[[files[i] for i in train_ind]]["num_comments"].values
    n_comments_dev = labels.set_index("source").loc[[files[i] for i in test_ind]]["num_comments"].values
    ## Isolate Feature Types We Care About
    LOGGER.info("Splicing Feature Set")
    vocabulary, X = update_vocabulary(vocabulary,
                                      X,
                                      use_text=USE_TEXT,
                                      use_subreddit=USE_SUBREDDIT,
                                      use_time=USE_TIME)
    ## Apply Dimensionality Reduction using Non-localness
    LOGGER.info("Performing Dimensionality Reduction")
    nl = Nonlocalness(vocabulary,
                      min_support=MIN_SUPPORT)
    text_nl_scores, text_agg_nl_scores = None, None
    sub_nl_scores, sub_agg_nl_scores = None, None
    if USE_TEXT:
        text_nl_scores, text_agg_nl_scores = nl.fit(X[train_ind], y_train_discrete, "text")
    if USE_SUBREDDIT:
        sub_nl_scores, sub_agg_nl_scores = nl.fit(X[train_ind], y_train_discrete, "subreddit")
    if USE_TEXT:
        X = nl.transform(X, "text", text_agg_nl_scores, TOP_K_TEXT)
    if USE_SUBREDDIT:
        X = nl.transform(X, "subreddit", sub_agg_nl_scores, TOP_K_SUBREDDITS)
    vocabulary = nl._vocabulary
    ## Fit Model
    LOGGER.info("Starting Model Fitting Procedure")
    geo = GeolocationInference(vocabulary)
    geo = geo.fit(X[train_ind], y_train)
    ## Make Predictions (Full Dataset)
    LOGGER.info("Making Predictions")
    coordinates = pd.DataFrame(y_train).drop_duplicates().values
    y_pred = geo.predict(X, coordinates)
    ## Compute Distance Error
    LOGGER.info("Computing Distance Error")
    dist_error_train = distance_between(y_train, y_pred[train_ind])
    dist_error_test = distance_between(y_test, y_pred[test_ind])
    ## Format Results
    LOGGER.info("Formating Results")
    train_res = pd.DataFrame(np.hstack([y_train, y_pred[train_ind], dist_error_train.reshape(-1,1), n_comments_train.reshape(-1,1)]),
                             columns=["longitude_true","latitude_true","longitude_pred","latitude_pred","error","num_comments"],
                             index=[files[i] for i in train_ind])
    dev_res = pd.DataFrame(np.hstack([y_test, y_pred[test_ind], dist_error_test.reshape(-1,1), n_comments_dev.reshape(-1,1)]),
                           columns=["longitude_true","latitude_true","longitude_pred","latitude_pred","error","num_comments"],
                           index=[files[i] for i in test_ind])
    return text_nl_scores, text_agg_nl_scores, sub_nl_scores, sub_agg_nl_scores, geo, train_res, dev_res

def run_fold(model_config,
             fold,
             split_dict,
             files,
             X,
             vocabulary,
             labels,
             output_dir):
    """
    Run one cross-validation fold (training and evaluation)

    Args:
        model_config (dict): Model training parameters
        fold (int): Which k-fold to consider
        split_dict (dict): Train/dev/test splits
        files (list): List of user files aligned with X
        vocabulary (Vocabulary): Learned vocabulary object
        labels (pandas DataFrame): User locations
        output_dir (str): Path to run output directory
    
    Returns:
        None, saves results to disk in respective fold directory
    """
    ## Create Output Directory
    fold_outdir = f"{output_dir}Fold_{fold}/"
    os.makedirs(fold_outdir)
    ## Create a Copy of The Vocabulary Class
    LOGGER.info("Copying Vocabulary Object")
    vocabulary = deepcopy(vocabulary)
    ## Identify Split Indices
    LOGGER.info("Identifying File Splits")
    train_files = set(split_dict["train"][fold])
    dev_files = set(split_dict["dev"][fold])
    train_ind = [i for i, f in enumerate(files) if f in train_files]
    dev_ind = [i for i, f in enumerate(files) if f in dev_files]
    ## Run Training Procedure
    text_nl_scores, text_agg_nl_scores, sub_nl_scores, sub_agg_nl_scores, geo, train_res, dev_res = run_training(
            model_config=model_config,
            train_files=train_files,
            test_files=dev_files,
            train_ind=train_ind,
            test_ind=dev_ind,
            files=files,
            labels=labels,
            X=X,
            vocabulary=vocabulary)
    ## Cache Results
    LOGGER.info("Caching Predictions")
    train_res["fold"] = fold
    dev_res["fold"] = fold
    train_res.to_csv(f"{fold_outdir}train_predictions.csv")
    dev_res.to_csv(f"{fold_outdir}dev_predictions.csv")
    ## Cache Other Objects (Model, Non-localness Calculations)
    LOGGER.info("Caching Model and Data Objects")
    if CACHE_MODELS:
        joblib.dump(geo, f"{fold_outdir}model.joblib")
    if model_config.get("USE_TEXT"):
        text_nl_scores.to_csv(f"{fold_outdir}nl_text.csv")
        text_agg_nl_scores.to_csv(f"{fold_outdir}nl_text_agg.csv", index=False)
    if model_config.get("USE_SUBREDDIT"):
        sub_nl_scores.to_csv(f"{fold_outdir}nl_subreddit.csv")
        sub_agg_nl_scores.to_csv(f"{fold_outdir}nl_subreddit_agg.csv", index=False)

def train_full_model(model_config,
                     split_dict,
                     files,
                     X,
                     vocabulary,
                     labels,
                     output_dir):
    """
    Train geolocation inference model on all available training data,
    test on held-out test set

    Args:
        model_config (dict): Model training parameters
        split_dict (dict): Cross-validation splits
        files (list): User filenames aligned with rows in X
        X (csr_matrix): Feature matrix
        vocabulary (Vocabulary): Learned vocabulary object
        labels (pandas DataFrame): User location labels
        output_dir (str): Path to run output directory
    """
    ## Create Output Directory
    model_outdir = f"{output_dir}Model/"
    os.makedirs(model_outdir)
    ## Create a Copy of The Vocabulary Class
    LOGGER.info("Copying Vocabulary Object")
    vocabulary = deepcopy(vocabulary)
    ## Identify Split Indices
    LOGGER.info("Identifying File Splits")
    test_files = set(split_dict["test"][0])
    train_files = set([i for i in files if i not in test_files])
    train_ind = [i for i, f in enumerate(files) if f in train_files]
    test_ind = [i for i, f in enumerate(files) if f in test_files]
    ## Run Training Procedure
    text_nl_scores, text_agg_nl_scores, sub_nl_scores, sub_agg_nl_scores, geo, train_res, test_res = run_training(
            model_config=model_config,
            train_files=train_files,
            test_files=train_ind,
            train_ind=train_ind,
            test_ind=test_ind,
            files=files,
            labels=labels,
            X=X,
            vocabulary=vocabulary)    
    ## Cache Results
    LOGGER.info("Caching Predictions")
    train_res["fold"] = 0
    test_res["fold"] = 0
    train_res.to_csv(f"{model_outdir}train_predictions.csv")
    test_res.to_csv(f"{model_outdir}test_predictions.csv")
    ## Cache Other Objects (Model, Non-localness Calculations)
    LOGGER.info("Caching Model and Data Objects")
    joblib.dump(geo, f"{model_outdir}model.joblib")
    if model_config.get("USE_TEXT"):
        text_nl_scores.to_csv(f"{model_outdir}nl_text.csv")
        text_agg_nl_scores.to_csv(f"{model_outdir}nl_text_agg.csv", index=False)
    if model_config.get("USE_SUBREDDIT"):
        sub_nl_scores.to_csv(f"{model_outdir}nl_subreddit.csv")
        sub_agg_nl_scores.to_csv(f"{model_outdir}nl_subreddit_agg.csv", index=False)


def main():
    """
    Run cross validation and evaluate on test data if desired.

    Args:
        None

    Returns:
        None
    """
    ## Parse Command Line
    args = parse_command_line()
    ## Load Settings, Configs
    settings = load_settings()
    data_config = load_data_config(args)
    model_config = load_model_config(args)
    ## Create Output Directory
    output_dir = create_cross_validation_directory(model_name=model_config.get("NAME"))
    LOGGER.info(f"Results being cached in: {output_dir}")
    ## Load Labels
    labels = load_labels(settings, data_config)
    ## Prepare Feature Set (or Load Cached)
    vocabulary, X, files, VOCAB_PARAMETERS, MIN_RESOLUTION, MIN_COMMENTS = prepare_feature_set(settings,
                                                                                               data_config,
                                                                                               labels)
    labels["num_comments"] = labels["num_comments"].map(lambda i: min(i, VOCAB_PARAMETERS["max_docs"]))
    ## Cache Cross Validation Parameters
    _ = cache_run_parameters(run_dir=output_dir,
                             data_config=data_config,
                             model_config=model_config)
    ## Filter Dataset By Desired Label Resolution
    labels, files, X = filter_dataset_by_resolution(model_config,
                                                    labels,
                                                    files,
                                                    X)
    ## Create Train/Dev/Test Splits
    split_dict = create_split_dict(model_config, 
                                   labels,
                                   files,
                                   output_dir)
    ## Run Cross-Validation (If not applying to test split)
    if not RUN_ON_TEST:
        for fold in range(K_FOLDS):
            tstart = datetime.now()
            LOGGER.info(f"Beginning Fold {fold+1}/{K_FOLDS} at {tstart.isoformat()}")
            _ = run_fold(model_config,
                         fold,
                         split_dict,
                         files,
                         X,
                         vocabulary,
                         labels,
                         output_dir)
            tstop = datetime.now()
            LOGGER.info("Finished Fold {} in {:.2f} seconds".format(fold+1, (tstop-tstart).total_seconds()))
    ## Train Model and Apply to Test Data
    else:
        tstart = datetime.now()
        LOGGER.info(f"Starting Model Training at {tstart.isoformat()}")
        _ = train_full_model(model_config,
                             split_dict,
                             files,
                             X,
                             vocabulary,
                             labels,
                             output_dir)
        tstop = datetime.now()
        LOGGER.info("Finished Training in {:.2f} seconds".format((tstop-tstart).total_seconds()))        
    
########################
### Execute
########################

if __name__ == "__main__":
    _ = main()
