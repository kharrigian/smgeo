
########################
### Imports
########################

## Standard Library
import os
import json
import gzip
import string

## External
import pytest

## Local
from smgeo.util.data_loaders import LoadProcessedData
from smgeo.util.tokenizer import (STOPWORDS,
                                  PRONOUNS)

########################
### Fixtures
########################

@pytest.fixture(scope="module")
def processed_data_file():
    """

    """
    ## Create Fake Data
    test_file = os.path.dirname(__file__) + "/../data/sample_processed_data.json.tar.gz"
    test_data = [
    {'user_id_str': 'FAKE_USER_ID_STR',
     'created_utc': 14142309301,
     'text': 'RT: I can\'t believe @FAKE_MENTION said this 💔 www.google.com',
     'text_tokenized': ['<RETWEET>', 'I', 'can', 'not_believe', '<USER_MENTION>', 'said', 'this', '💔', '<URL_TOKEN>', '<NEGATE_FLAG>'],
     'tweet_id': 'NOT_A_REAL_TWEET_ID_1',
     'depression': 'depression',
     'gender': 'F',
     'age': 31.20056089,
     'entity_type': 'tweet',
     'date_processed_utc': 1576219763,
     'source': 'FAKE_SOURCE_PATH.tweets.gz',
     'dataset': ['FAKE_DATASET']},
    {'user_id_str': 'FAKE_USER_ID_STR',
     'created_utc': 14143309301,
     'text': 'Ready to give up :( #Depression',
     'text_tokenized': ['Ready', 'to', 'give','up',':(', 'Depression'],
     'tweet_id': 'NOT_A_REAL_TWEET_ID_2',
     'depression': 'depression',
     'gender': 'F',
     'age': 31.20056089,
     'entity_type': 'tweet',
     'date_processed_utc': 1576219763,
     'source': 'FAKE_SOURCE_PATH.tweets.gz',
     'dataset': ['FAKE_DATASET']},     
    {'user_id_str': 'FAKE_USER_ID_STR',
     'created_utc': 14143309501,
     'text': 'NOT GOING TO SCHOOL TODAY. OR HOPEFULLY EVER AGAIN! PEACE 2019.',
     'text_tokenized': ['not_GOING', 'TO', 'SCHOOL', 'TODAY', '.', 'OR', 'HOPEFULLY', 'EVER', 'AGAIN', '!', 'PEACE', '<NUMERIC>', '.', '<NEGATE_FLAG>', '<UPPER_FLAG>'],
     'tweet_id': 'NOT_A_REAL_TWEET_ID_3',
     'depression': 'depression',
     'gender': 'F',
     'age': 31.20056089,
     'entity_type': 'tweet',
     'date_processed_utc': 1576219763,
     'source': 'FAKE_SOURCE_PATH.tweets.gz',
     'dataset': ['FAKE_DATASET']
    },    
    ]
    ## Save Test Data
    with gzip.open(test_file, "wt", encoding="utf-8") as the_file:
        json.dump(test_data, the_file)
    yield test_file, test_data
        ## Teardown
    _ = os.system(f"rm {test_file}")

@pytest.fixture(scope="module")
def processed_metadata_file():
    """

    """
    ## Create Fake Data
    test_metadata_file = os.path.dirname(__file__) + "/../data/sample_processed_data.meta.json.tar.gz"
    test_metadata = {'user_id_str': 'FAKE_USER_ID_STR',
                     'depression': 'depression',
                     'gender': 'F',
                     'age': 31.20056089,
                     'datasets': ['FAKE_DATASET'],
                     'split': 'train',
                     'num_tweets': 2888,
                     'num_words': 10032}
    ## Save Test Data
    with gzip.open(test_metadata_file, "wt", encoding="utf-8") as the_file:
        json.dump(test_metadata, the_file)
    yield test_metadata_file, test_metadata
    ## Teardown
    _ = os.system(f"rm {test_metadata_file}")

########################
### Globals
########################

## Initialization Parameters
DEFAULT_LOADER_INIT = {
    "filter_negate":False,
    "filter_upper":False,
    "filter_punctuation":False,
    "filter_numeric":False,
    "filter_user_mentions":False,
    "filter_url":False,
    "filter_retweet":False,
    "filter_stopwords":False,
    "keep_pronouns":True,
    "filter_empty":True,
    "emoji_handling":None,
    "max_tokens_per_document":None,
    "max_documents_per_user":None
}

########################
### Helpers
########################

def _check_filtered_out(raw_data,
                        filtered_data,
                        terms,
                        char_level=False):
    """

    """
    terms = set([i.lower() for i in terms])
    assert len(raw_data) == len(filtered_data)
    for raw, filtered in zip(raw_data, filtered_data):
        if not char_level:
            assert [i for i in raw["text_tokenized"] if i.lower() not in terms] == filtered["text_tokenized"]
        else:
            assert [i for i in raw["text_tokenized"] if not all(char.lower() in terms for char in i)] == filtered["text_tokenized"]
        for key in raw.keys():
            if key != "text_tokenized":
                assert key in filtered.keys()
                assert raw[key] == filtered[key]
    
########################
### Tests
########################

def test_flag_filtering(processed_data_file):
    """

    """
    ## Filter Term Flags to Check
    filter_terms = {
                    "filter_negate":(["<NEGATE_FLAG>"],False),
                    "filter_upper":(["<UPPER_FLAG>"],False),
                    "filter_punctuation":(string.punctuation,True),
                    "filter_numeric":(["<NUMERIC>"],False),
                    "filter_user_mentions":(["<USER_MENTION>"],False),
                    "filter_url":(["<URL_TOKEN>"],False),
                    "filter_retweet":(["<RETWEET>"],False)
                    }
    ## Get Test File and Data
    filename, raw_data = processed_data_file
    ## Check Each Filtering Independently
    for init_flag, (filter_set, char_level) in filter_terms.items():
        ## Initialize Loader
        loader_params = DEFAULT_LOADER_INIT.copy()
        loader_params[init_flag] = True
        loader = LoadProcessedData(**loader_params)
        ## Load Test Data
        filtered_data = loader.load_user_data(filename)
        ## Test Assertions
        _check_filtered_out(raw_data,
                            filtered_data,
                            filter_set,
                            char_level)


def test_stopword_filtering(processed_data_file):
    """

    """
    ## Get Test File and Data
    filename, raw_data = processed_data_file
    ## Cycle Through Pronoun Flag
    for keep_pronouns in [False, True]:
        ## Initialize Loader
        loader_params = DEFAULT_LOADER_INIT.copy()
        loader_params["filter_stopwords"] = True
        loader_params["keep_pronouns"] = keep_pronouns
        loader = LoadProcessedData(**loader_params)
        ## Load Test Data
        filtered_data = loader.load_user_data(filename)
        ## Initialize Expected Stopword Set
        stopset = set(STOPWORDS) | set([f"not_{w}" for w in STOPWORDS])
        if keep_pronouns:
            for l in list(stopset):
                if l in PRONOUNS:
                    stopset.remove(l)
        ## Check Filtering
        _check_filtered_out(raw_data, filtered_data, stopset, False)

def test_max_tokens_per_document(processed_data_file):
    """

    """
    ## Get Test File and Data
    filename, raw_data = processed_data_file
    ## Initialize Loader
    loader_params = DEFAULT_LOADER_INIT.copy()
    loader_params["max_tokens_per_document"] = 1
    loader = LoadProcessedData(**loader_params)
    ## Load Test Data
    filtered_data = loader.load_user_data(filename)
    ## Check
    assert len(filtered_data) == len(raw_data)
    for filtered in filtered_data:
        assert len(filtered["text_tokenized"]) == 1

def test_max_documents_per_user(processed_data_file):
    """

    """
    ## Get Test File and Data
    filename, raw_data = processed_data_file
    ## Initialize Loader
    loader_params = DEFAULT_LOADER_INIT.copy()
    loader_params["max_documents_per_user"] = 1
    loader = LoadProcessedData(**loader_params)
    ## Load Test Data
    filtered_data = loader.load_user_data(filename)
    ## Check
    assert len(filtered_data) == 1
    assert filtered_data[0]["tweet_id"] == "NOT_A_REAL_TWEET_ID_3"

def test_load_metadata(processed_metadata_file):
    """

    """
    ## Get Test File and Data
    filename, raw_data = processed_metadata_file
    ## Initialize Loader
    loader_params = DEFAULT_LOADER_INIT.copy()
    loader = LoadProcessedData(**loader_params) 
    ## Load Metadata
    metadata = loader.load_user_metadata(filename)
    ## Check Data Preserved
    assert raw_data == metadata

def test_emoji_handling(processed_data_file):
    """

    """
    ## Get Test File and Data
    filename, raw_data = processed_data_file
    ## Test 1: Strip
    loader_params = DEFAULT_LOADER_INIT.copy()
    loader_params["emoji_handling"] = "strip"
    loader = LoadProcessedData(**loader_params)
    replace_filtered = loader.load_user_data(filename)
    assert replace_filtered[0]["text_tokenized"] == \
           ['<RETWEET>', 'I', 'can', 'not_believe', '<USER_MENTION>', 'said', 'this', '<URL_TOKEN>', '<NEGATE_FLAG>']
    ## Test 2: Replace
    loader_params = DEFAULT_LOADER_INIT.copy()
    loader_params["emoji_handling"] = "replace"
    loader = LoadProcessedData(**loader_params)
    strip_filtered = loader.load_user_data(filename)
    assert strip_filtered[0]["text_tokenized"] == \
           ['<RETWEET>', 'I', 'can', 'not_believe', '<USER_MENTION>', 'said', 'this', '<EMOJI>', '<URL_TOKEN>', '<NEGATE_FLAG>']
    ## Test 3: Raise Error
    with pytest.raises(ValueError):
        loader = LoadProcessedData(emoji_handling="FAKE_ARGUMENT")
        loader.load_user_data(filename)

def test_preserve_case(processed_data_file):
    """

    """
    ## Get Test File and Data
    filename, raw_data = processed_data_file
    ## Test
    loader_params = DEFAULT_LOADER_INIT.copy()
    loader_params["preserve_case"] = False
    loader = LoadProcessedData(**loader_params)
    filtered_data = loader.load_user_data(filename)
    for f in filtered_data:
        assert all([not i.isupper() for i in f["text_tokenized"]])