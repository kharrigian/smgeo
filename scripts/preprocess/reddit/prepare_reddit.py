
#######################
### Configuration
#######################

## Directories
AUTHORS_RAW_DIR = "./data/raw/reddit/authors/"
AUTHORS_PROCESS_DIR = "./data/processed/reddit/authors/"

## Multiprocessing
NUM_PROCESSES = 8

## Tokenization Parameters
TOKENIZER_PARAMS = {
                    "stopwords":None,
                    "keep_case":False,
                    "negate_handling":True,
                    "negate_token":False,
                    "upper_flag":False,
                    "keep_punctuation":False,
                    "keep_numbers":False,
                    "expand_contractions":True,
                    "keep_user_mentions":False,
                    "keep_pronouns":True,
                    "keep_url":False,
                    "keep_hashtags":True,
                    "keep_retweets":False,
                    "emoji_handling":None
                   }

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
from smgeo.util.tokenizer import Tokenizer
from smgeo.util.helpers import flatten
from smgeo.util.logging import initialize_logger

#######################
### Globals
#######################

## Initialize Tokenizer
TOKENIZER = Tokenizer(**TOKENIZER_PARAMS)

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
    processed_author_data = []
    comment_times = []
    for comment in author_data:
        comment_time_utc = datetime.utcfromtimestamp(comment["created_utc"])
        comment_times.append(comment_time_utc)
        comment_processed = {
                    "text": [f"TOKEN={t}" for t in TOKENIZER.tokenize(comment["body"])],
                    "subreddit": "SUBREDDIT={}".format(comment["subreddit"].lower()),
                    "flair_text":[f"FLAIR_TOKEN={t}" for t in TOKENIZER.tokenize(comment["author_flair_text"])],
                    "created_utc":{
                                   "year":comment_time_utc.year,
                                   "month":comment_time_utc.month,
                                   "day":comment_time_utc.day,
                                   "hour":comment_time_utc.hour,
                                   "minute":comment_time_utc.minute
                                  },
        }
        processed_author_data.append(comment_processed)
    ## Sort Data (Newest Comments First)
    processed_author_data = list(map(lambda i: i[0], 
                                     sorted(zip(processed_author_data, comment_times), key = lambda x: x[1],
                                            reverse=True)))
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
