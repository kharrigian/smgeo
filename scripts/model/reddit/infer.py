
"""
Infer location for a list of reddit users using a pretrained inference model.
"""

#######################
### Imports
#######################

## Standard Library
import os
import sys
import json
import gzip
import argparse

## External Libraries
import joblib
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.sparse import vstack
import reverse_geocoder as rg # pip install reverse_geocoder

## Local Modules
from smgeo.acquire.reddit import RedditData
from smgeo.model import preprocess
from smgeo.util.logging import initialize_logger

## Logger
LOGGER = initialize_logger()

#######################
### Functions
#######################

def parse_command_line():
    """
    Parse command line for inference parameters

    Args:
        None
    
    Returns:
        args (Argparse object): Command-line arguments
    """
    ## Initialize Parser Object
    parser = argparse.ArgumentParser(description="Infer home location of reddit users using a pretrained model.")
    ## Required Arguments
    parser.add_argument("setting",
                        type=str,
                        choices={"retrieve","infer"},
                        help="'retrieve' if getting user data. 'infer' if user data exists already.")
    parser.add_argument("user_list",
                        type=str,
                        help="Path to .txt file containing users we want to infer location for")
    ## Optional Arguments
    parser.add_argument("--model_path",
                        type=str,
                        default=None,
                        help="Path to .joblib cached model file")
    parser.add_argument("--output_csv",
                        type=str,
                        default=None,
                        help="Path for saving the predictions. Should be a csv file.")
    parser.add_argument("--overwrite_existing_histories",
                        default=False,
                        action="store_true",
                        help="If this flag is specified, any existing histories will be requeried instead of used again.")
    parser.add_argument("--start_date",
                        default="2008-01-01",
                        type=str,
                        help="Start date for user history retrieval")
    parser.add_argument("--end_date",
                         default=None,
                         type=str,
                         help="End date for user history retrieval")
    parser.add_argument("--comment_limit",
                        default=250,
                        type=int,
                        help="Maximum number of comments to retrieve for each user. No guaranetee they are the most recent.")
    parser.add_argument("--min_comments",
                        default=0,
                        type=int,
                        help="Only make inferences who have at least this many comments in their post history. Users with more comments tend to have more accurate predictions.")
    parser.add_argument("--grid_cell_size",
                        default=.5,
                        type=float,
                        help="Coordinate grid cell size in degrees.")
    parser.add_argument("--posterior",
                        default=False,
                        action="store_true",
                        help="If this flag specified, the posterior across all coordinates will be included.")
    parser.add_argument("--reverse_geocode",
                        default=False,
                        action="store_true",
                        help="If this flag specified, the argmax of predictions will be reverse geocoded")
    parser.add_argument("--known_coordinates",
                         default=False,
                         action="store_true",
                         help="If specified, code will try to load in known training coordinates to restrict inference search space.")
    ## Parse Arguments
    args = parser.parse_args()
    ## Check That Required Files Exist
    if args.setting == "infer" and not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Could not find model: {args.model_path}")
    if not os.path.exists(args.user_list):
        raise FileNotFoundError(f"Could not find user list: {args.user_list}")
    if not args.user_list.endswith(".txt"):
        raise TypeError("Expected user list to be a .txt file with each user separated by a newline character")
    if args.setting == "infer" and not args.output_csv.endswith(".csv"):
        raise TypeError("Expected output_csv to be a .csv file")
    return args

def load_settings():
    """
    Load repository-wide settings parameters

    Args:
        None

    Returns:
        settings_config (dict): Repository-wide settings (e.g. data paths)
    """
    LOGGER.info("Loading Settings")
    settings_file =  os.path.dirname(os.path.abspath(__file__)) + \
                     "/../../../configurations/settings.json"
    if not os.path.exists(settings_file):
        raise FileNotFoundError(f"Could not find setting file in expected location: {settings_file}")
    with open(settings_file, "r") as the_file:
        settings_config = json.load(the_file)
    return settings_config

def load_users(args):
    """
    Load the names of Reddit users to infer locations for

    Args:
        args (Argparse object): Command-line arguments, includes user_list which
                                is a path to a txt file
    
    Returns:
        users (list of str): User names to infer location for
    """
    LOGGER.info("Loading User List")
    with open(args.user_list, "r") as the_file:
        users = the_file.readlines()
    users = [u.strip() for u in users]
    return users

def retrieve_user_data(args,
                       settings,
                       user_list):
    """
    Retrieve and cache comment history data for users

    Args:
        args (Argparse object): Command-line arguments
        settings (dict): Repository-wide settings
        user_list (list): List of users to infer locations for

    Returns:
        user_files (list): Paths to raw user files
    """
    LOGGER.info("Querying User Data")
    ## Cache Directory
    author_data_cache_dir = settings.get("reddit").get("AUTHORS_RAW_DIR")
    if not os.path.exists(author_data_cache_dir):
        os.makedirs(author_data_cache_dir)
    ## Intialize Reddit
    reddit = None
    ## Retrieve and Cache Directory
    user_files = []
    for user in tqdm(user_list, file=sys.stdout, total=len(user_list), desc="User History Request"):
        ## Check to See If User History has been Cached
        user_file = f"{author_data_cache_dir}{user}.json.gz"
        user_files.append(user_file)
        if os.path.exists(user_file) and not args.overwrite_existing_histories:
            continue
        ## Only need to initialize if user files don't already exist and not overwriting
        if reddit is None:
            reddit = RedditData(init_praw=False,
                                allow_praw=False,
                                warn_on_limit=True,
                                )
        ## Retrieve Comment Data
        user_comment_df = reddit.retrieve_author_comments(user, 
                                                          start_date=args.start_date,
                                                          end_date=args.end_date,
                                                          limit=args.comment_limit)
        ## Format Into Json Dictionary
        user_comment_list = preprocess.format_reddit_comments(user_comment_df)
        ## Cache
        with gzip.open(user_file, "wt") as the_file:
            json.dump(user_comment_list, the_file)
    return user_files

def prepare_data(model,
                 user_data_paths):
    """
    Format raw comment data into vectorized format

    Args:
        model (GeolocationInferenceModel): Trained inference model
        user_data_paths (list): List of raw data files
    
    Returns:
        X (csr_matrix): Feature matrix for application users
        n (list): Comment counts associated with each user
    """
    X = []
    n = []
    for user_file in tqdm(user_data_paths, total=len(user_data_paths), file=sys.stdout, desc="Vectorizing User Data"):
        ## Load Raw Data
        with gzip.open(user_file, "r") as the_file:
            user_data = json.load(the_file)
        ## Apply Data Preprocessing
        user_data_list = preprocess.process_reddit_comments(user_data)
        ## Data Filtering
        user_data_list = model._vocabulary._select_n_recent_documents(user_data_list)
        user_data_list = model._vocabulary._select_first_n_tokens(user_data_list)
        ## Add Number of Comments to Cache
        n.append(len(user_data_list))
        ## Count
        user_data_counts = model._vocabulary._count(user_data_list)
        ## Vectorize
        user_X = model._vocabulary._vectorize_user_data(user_data_counts)
        X.append(user_X)
    ## Stack
    X = vstack(X).tocsr()
    return X, n

def load_known_coordinates(settings):
    """
    Load coordinates used for training the original model (e.g. known cities)

    Args:
        settings (dict): Repository-wide settings (e.g. data paths)
    
    Returns:
        coordinates (2d-array): [Lon, Lat] coordinates to use for prediction
    """
    LOGGER.info("Loading Known Coordinates")
    author_training_file = "{}author_labels.json.gz".format(settings["reddit"]["LABELS_DIR"])
    if not os.path.exists(author_training_file):
        raise FileNotFoundError(f"Could not identify training data for loading known coordinates at: {author_training_file}. \
                                  Check placement of the label data or turn off the --known_coordinates flag.")
    labels = pd.read_json(author_training_file)
    coordinates = labels[["longitude","latitude"]].drop_duplicates().values
    return coordinates

def reverse_search(coordinates):
    """
    Use the Geonames Database to Reverse Search Locations based on Coordinates

    Args:
        coordinates (2d-array): [Lon, Lat] values
    
    Returns:
        result (list of dict): Reverse search results
    """
    result = rg.search(list(map(tuple,coordinates[:,::-1])))
    return result

def get_user_data_paths(users,
                        settings):
    """
    
    """
    ## Cache Directory
    author_data_cache_dir = settings.get("reddit").get("AUTHORS_RAW_DIR")
    if not os.path.exists(author_data_cache_dir):
        raise FileNotFoundError(f"Could not find author data directory: '{author_data_cache_dir}'")
    ## Files
    user_data_paths = []
    user_data_paths_missing = []
    for user in users:
        user_file = f"{author_data_cache_dir}/{user}.json.gz"
        if not os.path.exists(user_file):
            user_data_paths_missing.append(user)
        else:
            user_data_paths.append((user, user_file))
    if len(user_data_paths_missing) > 0:
        LOGGER.warning("WARNING: Missing data for the following users: {}".format(user_data_paths_missing))
    if len(user_data_paths) == 0:
        raise FileNotFoundError("Could not find any usable user data.")
    users = [u[0] for u in user_data_paths]
    data_paths = [u[1] for u in user_data_paths]
    return users, data_paths

def infer_user_data(args,
                    settings,
                    users):
    """
    
    """
    ## Load Geolocation Inference Model
    LOGGER.info("Loading Geolocation Inference Model")
    model = joblib.load(args.model_path)
    ## Get Updated Users and Data Paths
    LOGGER.info("Identifying Valid User Data Paths")
    users, user_data_paths = get_user_data_paths(users, settings)
    ## Prepare User Data
    LOGGER.info("Preparing User Data for Model")
    X, n = prepare_data(model, user_data_paths)
    ## Create Coordinate Grid
    LOGGER.info("Initializing Coordinate Grid")
    if not args.known_coordinates:
        coordinates = model._create_coordinate_grid(args.grid_cell_size)
    else:
        coordinates = load_known_coordinates(settings)
    ## Filter by Comment Size
    LOGGER.info("Filtering Users Based on Sample Size")
    X_mask = np.nonzero([i >= args.min_comments for i in n])[0]
    X = X[X_mask]
    users = [users[i] for i in X_mask]
    n = [n[i] for i in X_mask]
    ## Make Predictions
    LOGGER.info("Making Inferences")
    _, P = model.predict_proba(X, coordinates)
    y_pred = pd.DataFrame(index=users,
                          data=coordinates[P.argmax(axis=1)],
                          columns=["longitude_argmax","latitude_argmax"])
    ## Append Number of Comments Used to Make Predictions
    y_pred["n_comments"] = n
    ## Reverse Geocoding
    if args.reverse_geocode:
        LOGGER.info("Reversing the Geolocation Inferences")
        reverse = reverse_search(y_pred[["longitude_argmax","latitude_argmax"]].values)
        for level, level_name in zip(["name","admin2","admin1","cc"],["city","county","state","country"]):
            level_data = [i[level] for i in reverse]
            y_pred[f"{level_name}_argmax"] = level_data
    ## Add Posterior
    if args.posterior:
        LOGGER.info("Merging Posterior With Predictions")
        P = pd.DataFrame(P, index=users, columns=list(map(tuple, coordinates)))
        y_pred = pd.merge(y_pred, P, left_index=True, right_index=True)
    ## Cache
    LOGGER.info(f"Caching Inferences: {args.output_csv}")
    _ = y_pred.to_csv(args.output_csv, index=True)

def main():
    """
    Infer locations for a list of Reddit users

    Args:
        None
    
    Returns:
        None
    """
    ## Parse Command Line Arguments
    args = parse_command_line()
    ## Load Settings
    settings = load_settings()
    ## Load User List
    users = load_users(args)
    ## Stage 0: Retrieval
    if args.setting == "retrieve":
        ## Retrieve User Data
        _ = retrieve_user_data(args,
                               settings,
                               users)
    elif args.setting == "infer":
        _ = infer_user_data(args,
                            settings,
                            users)
    else:
        raise ValueError(f"Setting = '{args.setting}' not recognized.")
    ## Done
    LOGGER.info("Script complete.")

#######################
### Execute
#######################

if __name__ == "__main__":
    main()