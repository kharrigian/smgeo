
LABELS_DIR = "./data/raw/reddit/labels/"
RUN_GEOCODER = False

########################
### Imports
########################

## Standard Library
import os
import sys
import json
import gzip
from time import sleep
from datetime import datetime

## External Libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
from geopy.geocoders import GoogleV3
from scipy.spatial.distance import cdist, euclidean

## Local
from smgeo.util.location_extractor import LocationExtractor, country_continent_map
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
        return None
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

def flatten_geocoding_result(location_string,
                             region_bias,
                             res):
    """

    """
    ## Check Result
    if len(res) == 0:
        return {"query":location_string,
                "region_bias":region_bias}
    ## Create Copy
    res_copy = res.copy()
    res_copy["query"] = location_string
    res_copy["region_bias"] = region_bias
    ## Extract Address Components
    address_components = res_copy.pop("address_components", None)
    ordered_levels = ["locality",
                      "administrative_area_level_5",
                      "administrative_area_level_4",
                      "administrative_area_level_3",
                      "administrative_area_level_2",
                      "administrative_area_level_1",
                      "country",
                      "continent"]
    address_components_flat = {}
    for ol in ordered_levels:
        for ac in address_components:
            if ol in ac["types"]:
                address_components_flat[ol] = ac["long_name"]
    ## Drop Types and Add in Components
    _ = res_copy.pop("types", None)
    res_copy.update(address_components_flat)
    return res_copy


def geometric_median(X, eps=1e-5):
    """
    Source: https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points
    """
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1

def find_maximal_overlap(author, df, geo_flat_df):
    """

    """
    ## Geographic Levels in Order of Resolution (Subset of All)
    ordered_levels = ["locality",
                      "administrative_area_level_3",
                      "administrative_area_level_2",
                      "administrative_area_level_1",
                      "country",
                      "continent"]
    ## Isolate Author Data
    author_df = df.loc[df["author"]==author].drop_duplicates(subset=ordered_levels)
    ## Case 1: No Discrepencis
    if len(author_df) == 1:
        resolution = dict(map(lambda x: (x[0], x[1]) if not pd.isnull(x[1]) else (x[0], None), author_df.iloc[0][ordered_levels].items()))
        resolution["longitude"] = author_df.iloc[0]["longitude"]
        resolution["latitude"] = author_df.iloc[0]["latitude"]
        return resolution   
    ## Case 2: Need To Find Maximal Overlap
    resolution = dict((l, None) for l in ordered_levels)
    for o, ol in enumerate(ordered_levels[::-1]):
        if len(set(author_df[ol])) != 1:
            break
        resolution[ol] = author_df[ol].iloc[0]
    if o == 0:
        lon, lat = None, None
    else:
        rel_flat = geo_flat_df.copy()
        for ol in ordered_levels[::-1][:o]:
            if pd.isnull(resolution[ol]):
                rel_flat = rel_flat.loc[rel_flat[ol].isnull()]
            else:
                rel_flat = rel_flat.loc[rel_flat[ol] == resolution[ol]]
        author_res = pd.Series(resolution)[ordered_levels].isnull().idxmin()
        null_allowed = ordered_levels[:[i for i, o in enumerate(ordered_levels) if o == author_res][0]]
        null_before = rel_flat.loc[rel_flat[null_allowed].isnull().all(axis=1)]
        if len(null_before) > 0:
            lon, lat = null_before.iloc[0][["longitude","latitude"]].values
        else:
            lon, lat = geometric_median(rel_flat[["longitude","latitude"]].values)
        resolution["longitude"] = lon
        resolution["latitude"] = lat
    return resolution

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
seed_comments.dropna(subset=["author"], inplace=True)
seed_submissions.dropna(subset=["author"], inplace=True)

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
for loc_string, reg_bias in tqdm(location_strings, file=sys.stdout):
    if pd.isnull(reg_bias):
        reg_bias = None
        reg_bias_lbl = "null"
    else:
        reg_bias_lbl = reg_bias
    if loc_string in geocoding_results and reg_bias_lbl in geocoding_results[loc_string]:
        continue
    if loc_string not in geocoding_results:
        geocoding_results[loc_string] = {}
    if RUN_GEOCODER:
        geocoding_response =  geocode_string(loc_string, geocoder, reg_bias)
        if geocoding_response is not None:
            geocoding_results[loc_string][reg_bias_lbl] = geocoding_response
        sleep(geocoder_wait_time)

## Cache Results
if RUN_GEOCODER:
    with open(geocoding_results_file, "w") as the_file:
        json.dump(geocoding_results, the_file)

########################
### Clean Geocoded Strings
########################

## Flatten and Format Results into DataFrame
geocoding_results_flat = []
for locstring, region_dict in geocoding_results.items():
    for region, result in region_dict.items():
        geocoding_results_flat.append(flatten_geocoding_result(locstring, region, result))
geocoding_results_flat = pd.DataFrame(geocoding_results_flat)

## Merge Continent Where it Doesn't Exist
geocoding_results_flat["continent"] = geocoding_results_flat.apply(lambda row: country_continent_map[row["country"]] if row["country"] in country_continent_map else row["continent"], axis=1)

########################
### Merge Geocoding and Seed Data
########################

## Drop Rows With Too Few or Too Many Locations
merged_seed_data = merged_seed_data.loc[(merged_seed_data["locations"].map(len) > 0) &
                                        (merged_seed_data["locations"].map(len) < 3)].reset_index(drop=True).copy()

## Keep Only Most Recent Comment from Each Author
merged_seed_data["created_utc"] = merged_seed_data["created_utc"].astype(int)
merged_seed_data.sort_values("created_utc", ascending=True, inplace=True)
merged_seed_data = merged_seed_data.drop_duplicates("author", keep="last").reset_index(drop=True).copy()

## Flatten Merged Data Based on Location List
merged_seed_data_flat = []
for _, row in tqdm(merged_seed_data.iterrows(), total=len(merged_seed_data)):
    for location, bias in row["loc_region_biases"]:
        row_flat = row.drop(["loc_region_biases"]).to_dict()
        if pd.isnull(bias):
            bias = "null"
        geo_res = geocoding_results_flat.loc[(geocoding_results_flat["query"] == location)&
                                             (geocoding_results_flat["region_bias"]==bias)]
        if len(geo_res) == 0:
            continue
        row_flat.update(geo_res.iloc[0].to_dict())
        merged_seed_data_flat.append(row_flat)
merged_seed_data_flat = pd.DataFrame(merged_seed_data_flat)

########################
### Resolve Author Level Discrepancies
########################

## Compute Maximal Overlap
author_resolution_dict = {}
for author in tqdm(merged_seed_data_flat.author.unique(), file=sys.stdout):
    author_resolution_dict[author] = find_maximal_overlap(author, merged_seed_data_flat, geocoding_results_flat)
author_resolution_dict = pd.DataFrame.from_dict(author_resolution_dict, orient="index")

## Merge in Author Data
merged_cols = ["author",
               "body",
               "title", 
               "selftext",
               "created_utc",
               "id",
               "link_id",
               "locations",
               "subreddit",
               "source"]
author_resolution_merged = pd.merge(merged_seed_data_flat[merged_cols].drop_duplicates(subset=["author"]).set_index("author"),
                                    author_resolution_dict,
                                    left_index=True,
                                    right_index=True,
                                    how="right")

## Consolidate Text
author_resolution_merged["text"] = author_resolution_merged.apply(lambda row: row["title"] if row["source"]=="submissions" else row["body"], axis=1)
author_resolution_merged = author_resolution_merged.drop(["body","title","selftext"], axis=1)

## Drop Null Location Resoluton
author_resolution_merged.dropna(subset=["longitude"],inplace=True)
author_resolution_merged = author_resolution_merged.reset_index().rename(columns={"index":"author"})

## Dump Data
author_resolution_merged_json = []
for _, row in author_resolution_merged.iterrows():
    author_resolution_merged_json.append(json.loads(row.to_json()))
label_file = f"{LABELS_DIR}author_labels.json.gz"
with gzip.open(label_file, "wt") as the_file:
    json.dump(author_resolution_merged_json, the_file)