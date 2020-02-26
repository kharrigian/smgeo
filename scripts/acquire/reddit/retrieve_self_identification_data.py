
## Globals
DATA_DIR = "./data/raw/reddit/labels/"

#######################
### Imports
#######################

## Standard Library
from time import sleep
from datetime import datetime
import argparse

## External Libaries
import pandas as pd
from tqdm import tqdm

## Local Modules
from smgeo.acquire.reddit import RedditData
from smgeo.util.logging import initialize_logger

## Initialize Logger
logger = initialize_logger()

#######################
### Functions
#######################

def parse_arguments():
    """
    Parse command-line to identify configuration filepath.
    Args:
        None
    
    Returns:
        args (argparse Object): Command-line argument holder.
    """
    ## Initialize Parser Object
    parser = argparse.ArgumentParser(description="Collect location self-identification data")
    ## Generic Arguments
    parser.add_argument("--seed_submissions",
                        default=False,
                        action="store_true",
                        help="If included, retrieves comments from filtered seed submissions")
    parser.add_argument("--submission_titles",
                        default=False,
                        action="store_true",
                        help="If included, retrieves submissions from r/amateurroomporn")
    ## Parse Arguments
    args = parser.parse_args()
    ## Check Arguments
    if not args.seed_submissions and not args.submission_titles:
        raise ValueError("No data collection flags were specified")
    return args

def load_filtered_submissions(filtered_submission_file=f"{DATA_DIR}submission_candidates_filtered.csv"):
    """

    """
    ## Load Filtered Submissions
    filtered_submissions = pd.read_csv(filtered_submission_file)
    ## Drop Submissions to Ignore
    filtered_submissions = filtered_submissions.loc[filtered_submissions["ignore"].isnull()].reset_index(drop=True).copy()
    return filtered_submissions

def retrieve_submission_comments(filtered_submissions,
                                 max_retries=3,
                                 sleep_time=5):
    """

    """
    ## Get URLS
    urls = filtered_submissions["full_link"].tolist()
    ## Initialize Reddit Class
    reddit = RedditData(init_praw=True)
    ## Retrieve Comments
    comment_data = []
    failed_submissions = []
    for url in tqdm(urls, desc="Submissions", total=len(urls)):
        data_retrieved = False
        for _ in range(max_retries):
            try:
                comment_df = reddit.retrieve_submission_comments(url)
                comment_data.append(comment_df)
                data_retrieved = True
                break
            except:
                sleep(sleep_time)
        if not data_retrieved:
            failed_submissions.append(url)
    ## Concatenate Comment Data
    comment_data = pd.concat(comment_data)
    comment_data = comment_data.reset_index(drop=True).copy()
    return comment_data, failed_submissions
        
def retrieve_location_submissions(subreddits=["amateurroomporn"],
                                  start_date="2012-01-01",
                                  end_date="2020-02-01",
                                  limit=500000,
                                  max_retries=3,
                                  sleep_time=5):
    """

    """
    ## Separate Query Range by Month
    date_range = list(pd.date_range(start_date, end_date, freq="1M"))
    if pd.to_datetime(start_date) < date_range[0]:
        date_range = [pd.to_datetime(start_date)] + date_range
    if pd.to_datetime(end_date) > date_range[-1]:
        date_range = date_range + [pd.to_datetime(end_date)]
    date_range = list(map(lambda i: i.date().isoformat(), date_range))
    ## Initialize Reddit Class
    reddit = RedditData(init_praw=False)
    ## Get Submissions
    submissions = []
    for sub in subreddits:
        logger.info(f"Retrieving Submissions for r/{sub}")
        for start, stop in tqdm(zip(date_range[:-1], date_range[1:]), desc="Time Period", total = len(date_range) - 1):
            for _ in range(max_retries):
                try:
                    rng_submissions = reddit.retrieve_subreddit_submissions(sub,
                                                                            start,
                                                                            stop,
                                                                            limit)
                    if len(rng_submissions) > 0:
                        submissions.append(rng_submissions)
                    break
                except:
                    sleep(sleep_time)
    ## Concatenate Submissions
    submissions = pd.concat(submissions).reset_index(drop=True).copy()
    return submissions

def main():
    """

    """
    ## Parse Command Line Arguments
    args = parse_arguments()
    ## Run-date
    today = datetime.now().date().isoformat()
    ## Option 1: Seed Submissions
    if args.seed_submissions:
        ## Load Manually Curated Submission List
        filtered_submissions = load_filtered_submissions()
        ## Retrieve Comments
        comments, failures = retrieve_submission_comments(filtered_submissions)
        ## Log Failures
        for f in failures:
            logger.info(f"Failed to collect submission: {f}")
        ## Cache Comments
        comments.to_csv(f"{DATA_DIR}seed_submission_comments_{today}.csv", index=False)
    ## Option 2: Subreddit Submissions
    if args.submission_titles:
        ## Get Submissions
        submissions = retrieve_location_submissions(subreddits=["amateurroomporn"],
                                                    start_date="2012-01-01",
                                                    end_date="2020-02-01")
        ## Cache Submissions
        submissions.to_csv(f"{DATA_DIR}submission_titles_{today}.csv",index=False)
    ## Script Complete
    logger.info("Script Complete. Exiting.")

#######################
### Execute
#######################

if __name__ == "__main__":
    _ = main()