
"""
Geolocation Inference Training Script
"""

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

    """
    with open(args.data_config_path, "r") as the_file:
        data_config = json.load(the_file)
    return data_config

def load_model_config(args):
    """

    """
    with open(args.model_config_path, "r") as the_file:
        model_config = json.load(the_file)
    return model_config

def create_model_directory(settings,
                           model_config):
    """

    """
    ## Parameters
    MODELS_DIR = settings.get("reddit").get("MODELS_DIR")
    MODEL_NAME = model_config.get("NAME")
    ## Create Directory
    model_dir = f"{MODELS_DIR}{MODEL_NAME}/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        LOGGER.warn("Model directory already exists. Will overwrite existing files.")
    return model_dir

def cache_run_parameters(run_dir,
                         data_config,
                         model_config):
    """

    """
    config = {
        "data":data_config,
        "model":model_config,
        "runtime_utc":datetime.strftime(datetime.utcnow(), "%Y_%m_%d_%H_%M")
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


def load_labels(settings,
                data_config):
    """

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


def run_training(model_config,
                 files,
                 labels,
                 X,
                 vocabulary):
    """

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
    y_train = labels.set_index("source").loc[files][["longitude","latitude"]].values
    ## Get Discrete Labels (for Feature Selection)
    LOGGER.info("Discretizing Training Labels")
    labels_train_agg = aggregate_location_hierarchy(labels.loc[labels["source"].isin(set(files))],
                                                    min_support=10,
                                                    index_col="source",
                                                    ordered_levels=['administrative_area_level_1',
                                                                    'country',
                                                                    'continent'])
    y_train_discrete = labels_train_agg.loc[files].values
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
        text_nl_scores, text_agg_nl_scores = nl.fit(X, y_train_discrete, "text")
    if USE_SUBREDDIT:
        sub_nl_scores, sub_agg_nl_scores = nl.fit(X, y_train_discrete, "subreddit")
    if USE_TEXT:
        X = nl.transform(X, "text", text_agg_nl_scores, TOP_K_TEXT)
    if USE_SUBREDDIT:
        X = nl.transform(X, "subreddit", sub_agg_nl_scores, TOP_K_SUBREDDITS)
    vocabulary = nl._vocabulary
    ## Fit Model
    LOGGER.info("Starting Model Fitting Procedure")
    geo = GeolocationInference(vocabulary)
    geo = geo.fit(X, y_train)
    ## Return
    return text_nl_scores, text_agg_nl_scores, sub_nl_scores, sub_agg_nl_scores, geo


def main():
    """

    """
    ## Parse Command Line
    args = parse_command_line()
    ## Load Settings, Configs
    settings = load_settings()
    data_config = load_data_config(args)
    model_config = load_model_config(args)
    ## Create Output Directory
    model_dir = create_model_directory(settings, model_config)
    LOGGER.info(f"Model will be cached in: {model_dir}")
    ## Load Labels
    labels = load_labels(settings, data_config)
    ## Prepare Feature Set (or Load Cached)
    vocabulary, X, files, VOCAB_PARAMETERS, MIN_RESOLUTION, MIN_COMMENTS = prepare_feature_set(settings,
                                                                                               data_config,
                                                                                               labels)
    ## Cache Cross Validation Parameters
    _ = cache_run_parameters(run_dir=model_dir,
                             data_config=data_config,
                             model_config=model_config)
    ## Filter Dataset By Desired Label Resolution
    labels, files, X = filter_dataset_by_resolution(model_config,
                                                    labels,
                                                    files,
                                                    X)
    ## Train Model
    tstart = datetime.now()
    LOGGER.info(f"Starting Model Training at {tstart.isoformat()}")
    text_nl_scores, text_agg_nl_scores, sub_nl_scores, sub_agg_nl_scores, geo = run_training(model_config,
                                                                                             files,
                                                                                             labels,
                                                                                             X,
                                                                                             vocabulary)
    tstop = datetime.now()
    LOGGER.info("Finished Training in {:.2f} seconds".format((tstop-tstart).total_seconds()))
    ## Cache Objects (Model, Non-localness Calculations)
    LOGGER.info("Caching Model and Data Objects")
    joblib.dump(geo, f"{model_dir}model.joblib")
    if model_config.get("USE_TEXT"):
        text_nl_scores.to_csv(f"{model_dir}nl_text.csv")
        text_agg_nl_scores.to_csv(f"{model_dir}nl_text_agg.csv", index=False)
    if model_config.get("USE_SUBREDDIT"):
        sub_nl_scores.to_csv(f"{model_dir}nl_subreddit.csv")
        sub_agg_nl_scores.to_csv(f"{model_dir}nl_subreddit_agg.csv", index=False)

########################
### Execute
########################

if __name__ == "__main__":
    _ = main()
