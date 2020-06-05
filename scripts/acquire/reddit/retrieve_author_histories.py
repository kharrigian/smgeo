
LABELS_DIR = "./data/raw/reddit/labels/"
AUTHORS_DIR = "./data/raw/reddit/authors/"

##########################
### Imports
##########################

## Standard Library
import os
import sys
import gzip
import json
from datetime import datetime, timedelta

## External Library
from tqdm import tqdm
import pandas as pd

## Local
from smgeo.acquire.reddit import RedditData
from smgeo.util.logging import initialize_logger

LOGGER = initialize_logger()

##########################
### Functions
##########################

def get_threads_to_ignore():
    """
    Identify threads used for annotation that should not be included in
    any user's post history

    Args:
        None
    
    Returns:
        combined_links (set): Submission IDs to ignore
    """
    ## Load Queried Comment Data
    seed_comment_file = f"{LABELS_DIR}seed_submission_comments_2020-02-26.csv"
    seed_comments = pd.read_csv(seed_comment_file, low_memory=False)
    seed_comments.dropna(subset=["link_id"],inplace=True)
    ## Load Queried Submission Titles
    seed_submission_file = f"{LABELS_DIR}submission_titles_2020-02-26.csv"
    seed_submissons = pd.read_csv(seed_submission_file, low_memory=False)
    seed_submissons.dropna(subset=["id"],inplace=True)
    ## Links to Ignore
    comment_links = set(seed_comments.loc[seed_comments.link_id.str.contains("_")]["link_id"].str.split("_").map(lambda i: i[1]))
    submission_links = set(seed_submissons["id"])
    combined_links = comment_links | submission_links
    return combined_links

def query_author_data(reddit,
                      author,
                      label_source_date,
                      link_ids_to_ignore):
    """
    Query comment history for a single user. Considers at most
    90 days after and 2 years before the user's annotation post

    Args:
        reddit (RedditData): Library API wrapper
        author (str): Name of reddit user
        label_source_date (int): Date the comment used for annotation came from
        link_ids_to_ignore (set): Submission IDs to filter out from comment data
    
    Returns:
        None, dumps JSON data to disk
    """
    ## Establish Write File
    author_file = f"{AUTHORS_DIR}{author}.json.gz"
    if os.path.exists(author_file):
        return
    ## Get Time Range
    label_source_date = reddit.convert_utc_epoch_to_datetime(label_source_date).date()
    end_date = min(datetime.now().date(), label_source_date + timedelta(90))
    start_date = label_source_date - timedelta(365 * 2)
    ## Query Data
    author_data = reddit.retrieve_author_comments(author,
                                                  start_date=start_date.isoformat(),
                                                  end_date=end_date.isoformat())
    ## Clean
    if author_data is None or len(author_data) == 0:
        author_data_json = []
    else:
        ## Filter Out Threads Used as Part of Labeling
        author_data = author_data.loc[~author_data.link_id.map(lambda i: i.split("_")[1] in link_ids_to_ignore if isinstance(i, str) else False)]
        ## Format as Json Dictionaries
        author_data_json = []
        for _, row in author_data.iterrows():
            author_data_json.append(json.loads(row.to_json()))
    ## Cache Data
    with gzip.open(author_file, "wt") as the_file:
        json.dump(author_data_json, the_file)


def main():
    """
    Query author history in serial. Save raw
    data to disk as Gzipped JSON lists

    Args:
        None
    
    Returns:
        None
    """
    ## Output Directory
    if not os.path.exists(AUTHORS_DIR):
        os.makedirs(AUTHORS_DIR)
    ## Load User Labels
    LOGGER.info("Loading Labels")
    labels = pd.read_json(f"{LABELS_DIR}author_labels.json.gz")
    ## Identify Threads to Ignore When Querying Data
    LOGGER.info("Identifying Threads to Ignore")
    link_ids_to_ignore = get_threads_to_ignore()
    ## Query Data
    LOGGER.info("Starting Author Data Query")
    reddit = RedditData()
    for (author, label_source_date) in tqdm(labels[["author","created_utc"]].values, file=sys.stdout):
        _ = query_author_data(reddit, author, label_source_date, link_ids_to_ignore)
    LOGGER.info("Script Complete.")

##########################
### Execute
##########################

if __name__ == "__main__":
    _ = main()