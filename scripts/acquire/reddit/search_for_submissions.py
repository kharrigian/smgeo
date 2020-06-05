
####################
### Imports
####################

## Standard Library
import os
import time

## External Library
import pandas as pd
from tqdm import tqdm

## Local Modules
from smgeo.acquire.reddit import RedditData
from smgeo.util.logging import initialize_logger

####################
### Globals
####################

## Query Params
MIN_COMMENTS = 25
START_DATE = "2005-08-01"
END_DATE="2020-01-20"

## Sleep Time
API_SLEEP_TIME = 2

## Logging
LOGGER = initialize_logger()

####################
### Functions
####################

def get_query_groups(start_date="2005-08-01",
                     end_date="2020-01-20",
                     query_freq="3M",
                     query_limit=100000):
    """
    Create a set of dictonaries specifying submission query terms

    Args:
        start_date (str): ISO-format datetime for starting the search
        end_date (str): ISO-format datetime for ending the search
        query_freq (str): Time to consider per query
        query_limit (int): Maximum number of results to return
    
    Returns:
        query_groups (list of dict): Title query parameter list
    """
    ## Search Terms
    search_terms = [
        "where are you from",
        "where do you live",
        "where are you living"
    ]
    ## Date Range
    drange = list(pd.date_range(start_date, end_date, freq=query_freq))
    if drange[0] > pd.to_datetime(start_date):
        drange = [pd.to_datetime(start_date)] + drange
    if drange[-1] < pd.to_datetime(end_date):
        drange = drange + [pd.to_datetime(end_date)]
    ## Get Query Groups
    query_groups = []
    for start, stop in zip(drange[:-1], drange[1:]):
        for st in search_terms:
            query = {"query":st,
                     "subreddit":None,
                     "start_date":start.date().isoformat(),
                     "end_date":stop.date().isoformat(),
                     "limit":query_limit}
            query_groups.append(query)
    return query_groups

def filter_submission_results(submission_results,
                              min_comments=50):
    """
    Drop submission results that are duplicates or do not meet
    a minimum comment size

    Args:
        submission_results (pandas DataFrame): Submission metadata DF
        min_comments (int): Minimum number of unique comments to keep a 
                            submission
    
    Returns:
        submission_results (pandas DataFrame): Filtered results
    """
    ## Drop Duplicates
    submission_results = submission_results.drop_duplicates(subset=["id"])
    ## Filter by Minimum Commments
    submission_results = submission_results.loc[submission_results["num_comments"] >= min_comments]
    ## Reset Index
    submission_results = submission_results.reset_index(drop=True).copy()
    return submission_results

def main():
    """
    Query submissions to use for manual filtering.

    Args:
        None
    
    Returns:
        None
    """
    ## Initialize API
    r = RedditData(False)
    ## Get Query Groups
    LOGGER.info("Getting Query Groups")
    query_groups = get_query_groups(start_date=START_DATE,
                                    end_date=END_DATE)
    ## Search for Submissions
    LOGGER.info("Searching for Relevant Submissions")
    submission_results = []
    for q in tqdm(query_groups, desc="Submission Queries"):
        q_df = r.search_for_submissions(**q)
        if q_df is not None and len(q_df) > 0:
            submission_results.append(q_df)
        time.sleep(API_SLEEP_TIME)
    submission_results = pd.concat(submission_results)
    submission_results = submission_results.reset_index(drop=True).copy()
    ## Filtering
    LOGGER.info("Filtering Submissions")
    submission_results_filtered = filter_submission_results(submission_results,
                                                            min_comments=MIN_COMMENTS)
    ## Subset and Sort
    LOGGER.info("Sorting and Subseting Submissions")
    cols = ["title","author","created_utc","full_link","id","num_comments"]
    submission_results_filtered = submission_results_filtered[cols].sort_values("num_comments",ascending=False)
    submission_results_filtered = submission_results_filtered.reset_index(drop=True).copy()
    ## Cache
    LOGGER.info("Caching Submission Candidates")
    outfolder = "./data/raw/reddit/labels/"
    outfile = f"{outfolder}submission_candidates.csv"
    if not os.path.exists(outfile):
        os.makedirs(outfolder)
    submission_results_filtered.to_csv(outfile, index=False)
    LOGGER.info("Done.")

####################
### Execution
####################

if __name__ == "__main__":
    _ = main()
