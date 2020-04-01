
#######################
### Configuration
#######################

## Directories
AUTHORS_RAW_DIR = "./data/raw/reddit/authors/"
AUTHORS_PROCESS_DIR = "./data/processed/reddit/authors/"

## Multiprocessing
NUM_PROCESSES = 8

## Re-run Processing on Existing data
RERUN_PROCESSING = False

#######################
### Imports
#######################

## Standard Library
import os
import sys
import json
import gzip
from glob import glob
from datetime import datetime
from collections import Counter
from multiprocessing import Pool

## External Libraries
from tqdm import tqdm

## Local Modules
from smgeo.model.preprocess import process_reddit_comments
from smgeo.util.logging import initialize_logger

#######################
### Globals
#######################

## Logging 
LOGGER = initialize_logger()

#######################
### Functions
#######################

def process_author_data(filename):
    """

    """
    ## Identify Author Name
    author = os.path.basename(filename).split(".json.gz")[0]
    ## Output Filename
    author_outfile = f"{AUTHORS_PROCESS_DIR}{author}.comments.json.gz"
    author_meta_outfile = f"{AUTHORS_PROCESS_DIR}{author}.meta.json.gz"
    ## If Files Exist, No Processing Needed
    if os.path.exists(author_outfile) and os.path.exists(author_meta_outfile) and not RERUN_PROCESSING:
        return None
    ## Load Author Data
    with gzip.open(filename, "r") as the_file:
        author_data = json.load(the_file)
    ## No Data
    if len(author_data) == 0:
        ## Cache Comments
        with gzip.open(author_outfile, "wt") as the_file:
            json.dump([], the_file)
        ## Meta
        author_metadata = {
                        "author":author,
                        "num_comments": 0,
                        "num_tokens":0,
                        "processed_time_utc":datetime.utcnow().isoformat(),
                        "subreddits":{},
                        "created_utc_max":None,
                        "created_utc_min":None
                        }
        ## Cache Metadata
        with gzip.open(author_meta_outfile, "wt") as the_file:
            json.dump(author_metadata, the_file)
        return
    ## Process Author Data
    processed_author_data = process_reddit_comments(author_data)
    ## Cache Comments
    with gzip.open(author_outfile, "wt") as the_file:
        json.dump(processed_author_data, the_file)
    ## Get Metadata
    author_metadata = {
        "author":author,
        "num_comments": len(processed_author_data),
        "num_tokens":sum([len(i["text"]) for i in processed_author_data]),
        "processed_time_utc":datetime.utcnow().isoformat(),
        "subreddits":Counter([i["subreddit"] for i in processed_author_data]),
        "created_utc_max":processed_author_data[0]["created_utc"],
        "created_utc_min":processed_author_data[-1]["created_utc"]
    }
    ## Cache Metadata
    with gzip.open(author_meta_outfile, "wt") as the_file:
        json.dump(author_metadata, the_file)


def main():
    """

    """
    ## Setup Output Directory
    if not os.path.exists(AUTHORS_PROCESS_DIR):
        os.makedirs(AUTHORS_PROCESS_DIR)
    ## Identify Raw Author Files
    raw_author_files = glob(f"{AUTHORS_RAW_DIR}*")
    ## Process Data
    mp = Pool(NUM_PROCESSES)
    _ = list(tqdm(mp.imap_unordered(process_author_data, raw_author_files), total = len(raw_author_files), file=sys.stdout))
    mp.close()
    ## Done
    LOGGER.info("Script Complete")

#######################
### Execute
#######################

if __name__ == "__main__":
    main()
