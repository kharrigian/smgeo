
##########################
### Imports
##########################

## Standard Library
import json
from datetime import datetime

## External Libraries
from pandas import DataFrame

## Local Modules
from ..util.tokenizer import Tokenizer

###########################
### Globals
###########################

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

## Initialize Tokenizer
TOKENIZER = Tokenizer(**TOKENIZER_PARAMS)

###########################
### Functions
###########################

def format_reddit_comments(comment_df):
    """
    Convert raw Reddit comment data (pulled from the acquire module)
    into a list of JSON dictionaries to be processed

    Args:
        comment_df (pandas DataFrame): Comment dataframe
    
    Returns:
        comment_list (list of dict): JSON dictionaries of comment data
    """
    ## Data Check
    if comment_df is None or len(comment_df) == 0:
        return []
    if not isinstance(comment_df, DataFrame):
        raise TypeError("Expected comment_df to be a pandas DataFrame.")
    ## Convert to JSON
    comment_list = []
    for _, row in comment_df.iterrows():
        comment_list.append(json.loads(row.to_json()))
    return comment_list

def process_reddit_comments(comment_list):
    """
    Tokenize and prepare raw input features from list of comments

    Args:
        comment_list (list of dict): JSON dictionaries of comment data

    Returns:
        processed_comment_list (list of dict): Tokenized and featurized comment dictionaries,
                                            sorted from newest to oldest
    """
    ## Process Author Data
    processed_comment_list = []
    comment_times = []
    for comment in comment_list:
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
        processed_comment_list.append(comment_processed)
    ## Sort Data (Newest Comments First)
    processed_comment_list = list(map(lambda i: i[0], 
                                      sorted(zip(processed_comment_list, comment_times), key = lambda x: x[1],
                                                 reverse=True)))
    return processed_comment_list
