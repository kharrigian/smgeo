
LABELS_DIR = "./data/raw/reddit/labels/"

########################
### Imports
########################

## Standard Library
import os
import json
from time import sleep
from datetime import datetime

## External Libraries
import pandas as pd
from tqdm import tqdm
from geopy.geocoders import GoogleV3

## Local
from smgeo.util.location_extractor import LocationExtractor
from smgeo.util.helpers import flatten
from smgeo.util.logging import initialize_logger

LOGGER = initialize_logger()

########################
### Helpers
########################

def _initialize_google_geocoder(config_file="config.json"):
    """

    """
    ## Load Credentials
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Could not find configuration file: `{config_file}`")
    with open(config_file, "r") as the_file:
        config = json.load(the_file)
    ## Get Google Credentials
    if "google" not in config:
        raise KeyError("Configuration should include `google` credentials")
    config = config.get("google")
    ## Initialize Geocoder
    geocoder = GoogleV3(api_key=config.get("api_key"))
    return geocoder

def geocode_string(query, geocoder, region=None):
    """

    """
    ## Make Request
    query_time_utc = datetime.utcnow().isoformat()
    try:
        response = geocoder.geocode(query=query,
                                    exactly_one=True,
                                    region=region,
                                    language="english")
    except Exception as e:
        LOGGER.info(f"Encountered error for query: {query}")
        raise(e)
        return {}
    ## Parse Response
    response_dict = {}
    if response is not None:
        response_dict = {
                    "query_time_utc":query_time_utc,
                    "address":response.address,
                    "latitude":response.latitude,
                    "longitude":response.longitude,
                    "types":response.raw["types"],
                    "place_id":response.raw["place_id"],
                    "address_components":response.raw["address_components"]
        }
    return response_dict

########################
### Load Seed Data
########################

## Comment Data
seed_comment_file = f"{LABELS_DIR}seed_submission_comments_2020-02-26.csv"
seed_comments = pd.read_csv(seed_comment_file, low_memory=False)

## Submission Titles
seed_submission_file = f"{LABELS_DIR}submission_titles_2020-02-26.csv"
seed_submissions = pd.read_csv(seed_submission_file, low_memory=False)

########################
### Initial Filtering
########################

## Drop Comments Without Text
seed_comments = seed_comments.dropna(subset=["body"])

## Top-level Comments Only
seed_comments = seed_comments.loc[seed_comments["parent_id"] == seed_comments["link_id"]]

## Transient User Filter
ignore_terms = ["move","moving","born","raised","travel","trip","leave","leaving"]
seed_comments = seed_comments.loc[seed_comments["body"].str.lower().map(lambda x: not any(i in x for i in ignore_terms))]
seed_submissions = seed_submissions.loc[seed_submissions["title"].str.lower().map(lambda x: not any(i in x for i in ignore_terms))]

## Filter Out Deleted and Removed Comments/Users
seed_comments = seed_comments.loc[~seed_comments["body"].isin(set(["[deleted]","[removed]"]))]
seed_comments = seed_comments.loc[seed_comments["author"]!="[deleted]"]
seed_submissions = seed_submissions.loc[seed_submissions["author"]!="[deleted]"]

## Reset Index
seed_comments = seed_comments.reset_index(drop=True)
seed_submissions = seed_submissions.reset_index(drop=True)

########################
### Location String Identification
########################

## Initialize Extractor
extractor = LocationExtractor()

## Look For Matches
seed_comments["locations"] = seed_comments["body"].map(extractor.find_locations)
seed_submissions["locations"] = seed_submissions["title"].map(extractor.find_locations)

########################
### Prepare/Load Subreddit Biases
########################

## Subreddit File
subreddit_bias_file = f"{LABELS_DIR}subreddit_biases.csv"
if not os.path.exists(subreddit_bias_file):
    subreddit_biases = seed_comments[["subreddit"]].drop_duplicates()
    subreddit_biases["region"] = None
    subreddit_biases.to_csv(subreddit_bias_file, index=False)
    LOGGER.info("Dumping subreddits for automatic labeling. Please label and then re-run.")
    exit

## Load Biases
subreddit_biases = pd.read_csv(subreddit_bias_file)

########################
### Geocoding
########################

## Consolidate Sources
seed_comments["source"] = "comments"
seed_submissions["source"] = "submissions"
merged_seed_data = pd.concat([seed_comments[["author","body","created_utc","id","link_id","subreddit","locations","source"]],
                              seed_submissions[["author","title","selftext","created_utc","id","subreddit","locations","source"]]],
                              sort=True)

## Append Region Biases
merged_seed_data = pd.merge(merged_seed_data,
                            subreddit_biases,
                            on=["subreddit"],
                            how="left")

## Get Unique String, Bias Combinations
merged_seed_data["loc_region_biases"] = merged_seed_data.apply(lambda row: [(place,row["region"]) for place in row["locations"]], axis=1)
location_strings = sorted(set(flatten(merged_seed_data["loc_region_biases"])), key=lambda x: x[0])

## Initialize Geocoder
geocoder = _initialize_google_geocoder("config.json")

## Initialize/Load Cache of Results
geocoding_results_file = f"{LABELS_DIR}google_geocoder_results.json"
geocoding_results = {}
if os.path.exists(geocoding_results_file):
    with open(geocoding_results_file, "r") as the_file:
        geocoding_results = json.load(the_file)

## Cycle Through Strings to Geocode
geocoder_wait_time = 0.01
for loc_string, reg_bias in tqdm(location_strings):
    if pd.isnull(reg_bias):
        reg_bias = None
        reg_bias_lbl = "null"
    else:
        reg_bias_lbl = reg_bias
    if loc_string in geocoding_results and reg_bias_lbl in geocoding_results[loc_string]:
        continue
    if loc_string not in geocoding_results:
        geocoding_results[loc_string] = {}
    geocoding_results[loc_string][reg_bias_lbl] = geocode_string(loc_string, geocoder, reg_bias)
    sleep(geocoder_wait_time)

## Cache Results
with open(geocoding_results_file, "w") as the_file:
    json.dump(geocoding_results, the_file)

