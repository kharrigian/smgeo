
LABELS_DIR = "./data/raw/reddit/labels/"

########################
### Imports
########################

## Standard Library

## External Libraries
import pandas as pd

## Local
from smgeo.util.location_extractor import LocationExtractor
from smgeo.util.helpers import flatten

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
