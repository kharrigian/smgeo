
#######################
### Imports
#######################

## Standard
import os
import pytz
import json
from datetime import datetime

## External Libraries
import pytest
import pandas as pd

## Local
from smgeo.acquire.reddit import RedditData

#######################
### Fixtures
#######################

@pytest.fixture(scope="module")
def reddit_psaw():
    """

    """
    ## Initialize With PSAW
    reddit = RedditData(init_praw=False)
    return reddit

@pytest.fixture(scope="module")
def reddit_praw():
    """

    """
    ## Initialize with PRAW
    try:
        reddit = RedditData(init_praw=True)
    except FileNotFoundError:
        reddit = MockRedditDataPraw()
    ## Check Credentials are Valid
    try:
        reddit.api.r.user.me()
    except Exception as e:
        if e.response.json()["message"] == "Unauthorized":
            reddit = MockRedditDataPraw()
    return reddit

#######################
### Mocking
#######################

class MockRedditDataPraw(RedditData):

    def __init__(self):
        """

        """
        ## Inherit Existing Methods
        super(MockRedditDataPraw, self).__init__(False)
        ## Update API
        self.api.r = 1

#######################
### Tests
#######################

def test_repr(reddit_psaw):
    """

    """
    reddit_repr = reddit_psaw.__repr__()
    assert reddit_repr == "RedditData(init_praw=False)"

def test_init_psaw_wrapper(reddit_psaw):
    """

    """
    assert hasattr(reddit_psaw, "api")
    assert reddit_psaw.api.r is None

def test_init_praw_wrapper(reddit_praw):
    """

    """
    assert hasattr(reddit_praw, "api")
    assert reddit_praw.api.r is not None

def test_get_start_date(reddit_psaw):
    """

    """
    ## No Start Date
    no_start_epoch = reddit_psaw._get_start_date(None)
    ## Provided Start Date
    default_start_epoch = reddit_psaw._get_start_date("2005-08-01")
    ## Check
    assert no_start_epoch == default_start_epoch
    assert no_start_epoch == int(datetime(*[2005,8,1]).timestamp())

def test_get_end_date(reddit_psaw):
    """

    """
    ## No End Date
    no_end_date = reddit_psaw._get_end_date(None)
    ## Get Tomorrow
    now = datetime.now().date()
    tomorrow = datetime(now.year, now.month, now.day+1)
    ## Tomorrow End Date
    tomorrow_end_date = reddit_psaw._get_end_date(tomorrow.date().isoformat())
    ## Tests
    tomorrow_end_date_expected = int(tomorrow.timestamp())
    assert no_end_date == tomorrow_end_date == tomorrow_end_date_expected

def test_retrieve_subreddit_submissions(reddit_psaw,
                                        reddit_praw):
    """

    """
    ## Params
    params = {"subreddit":"modeltrains",
              "start_date":"2019-01-01",
              "end_date":"2019-01-02",
              "limit":10}
    ## Get Submissions from Both
    sub_psaw = reddit_psaw.retrieve_subreddit_submissions(**params)
    sub_praw = reddit_praw.retrieve_subreddit_submissions(**params)
    ## Tests
    assert isinstance(sub_psaw, pd.DataFrame)
    assert isinstance(sub_praw, pd.DataFrame)
    assert sub_praw.columns.tolist() == sub_praw.columns.tolist()
    assert sub_praw.shape == sub_psaw.shape
    assert set(sub_praw.created_utc) == set(sub_psaw.created_utc)
    assert sub_praw.created_utc.tolist() == sorted(sub_praw.created_utc.values)

def test_retrieve_submission_comments(reddit_psaw,
                                      reddit_praw):
    """

    """
    ## Sample Submission (Equivalent)
    sub1 = "https://www.reddit.com/r/modeltrains/comments/6v7yvh/layout_update_and_a_question_see_comments/"
    sub2 = "t3_6v7yvh"
    ## Get Data
    df1_psaw = reddit_psaw.retrieve_submission_comments(sub1)
    df2_psaw = reddit_psaw.retrieve_submission_comments(sub2)
    df1_praw = reddit_praw.retrieve_submission_comments(sub1)
    df2_praw = reddit_praw.retrieve_submission_comments(sub2)
    ## Tests
    for df in [df1_psaw, df1_praw, df2_psaw, df2_praw]:
        assert isinstance(df, pd.DataFrame)
    assert (df1_praw.fillna("") == df2_praw.fillna("")).all().all()
    assert (df1_psaw.fillna("") == df2_psaw.fillna("")).all().all()
    assert df1_praw.shape == df1_psaw.shape == df2_praw.shape == df2_psaw.shape
    assert df1_psaw.created_utc.tolist() == sorted(df1_psaw.created_utc.values)

def test_retrieve_author_comments(reddit_psaw,
                                  reddit_praw):
    """

    """
    ## Params
    params = {"author":"HuskyKeith",
              "start_date":"2019-12-01",
              "end_date":"2019-12-31",
              "limit":None}
    ## Make Request
    com_psaw = reddit_psaw.retrieve_author_comments(**params)
    com_praw = reddit_praw.retrieve_author_comments(**params)
    ## Test
    assert isinstance(com_psaw, pd.DataFrame)
    assert isinstance(com_praw, pd.DataFrame)
    assert com_psaw.shape == com_praw.shape
    if isinstance(reddit_praw.api.r, int):
        assert com_praw.body.values[0] == com_psaw.body.values[0]
    else:
        assert com_praw.body.values[0] != com_psaw.body.values[0] # Edited typo in comment
    len(com_praw.body.values[0].split()) == len(com_psaw.body.values[0].split())
    assert com_psaw.author.values[0] == com_praw.author.values[0] == "HuskyKeith"

def test_retrieve_author_submissions(reddit_psaw,
                                     reddit_praw):
    """

    """
    ## Params
    params = {"author":"HuskyKeith",
              "start_date":"2018-05-25",
              "end_date":"2018-05-27",
              "limit":None}
    ## Make Request
    sub_psaw = reddit_psaw.retrieve_author_submissions(**params)
    sub_praw = reddit_praw.retrieve_author_submissions(**params)
    ## Test
    assert isinstance(sub_psaw, pd.DataFrame)
    assert isinstance(sub_praw, pd.DataFrame)
    assert len(sub_praw) > 0
    assert sub_praw.shape == sub_psaw.shape
    assert sub_praw["title"].item() == sub_psaw["title"].item()

def test_search_for_submissions(reddit_psaw,
                                reddit_praw):
    """

    """
    ## Params
    params = {"query":"Mixed Effects Linear Model",
              "subreddit":"AskStatistics",
              "start_date":"2018-05-25",
              "end_date":"2018-05-27",
              "limit":1}
    ## Make Requests
    sub_psaw = reddit_psaw.search_for_submissions(**params)
    sub_praw = reddit_praw.search_for_submissions(**params)
    ## Tests
    assert isinstance(sub_psaw, pd.DataFrame)
    assert isinstance(sub_praw, pd.DataFrame)
    assert len(sub_praw) == 1
    assert sub_praw.shape == sub_psaw.shape
    assert sub_praw["title"].item() == sub_psaw["title"].item()
    assert sub_praw["author"].item() == "HuskyKeith"

def test_search_for_comments(reddit_psaw,
                             reddit_praw):
    """

    """
    ## Params
    params = {"query":"Geonames",
              "subreddit":"LanguageTechnology",
              "start_date":"2019-12-22",
              "end_date":"2019-12-25",
              "limit":1}
    ## Make Request
    com_psaw = reddit_psaw.search_for_comments(**params)
    com_praw = reddit_praw.search_for_comments(**params)
    ## Tests
    assert isinstance(com_psaw, pd.DataFrame)
    assert isinstance(com_praw, pd.DataFrame)
    assert len(com_praw) == 1
    assert com_praw.shape == com_psaw.shape
    assert com_praw["author"].item() == "HuskyKeith"
    assert com_praw["link_id"].item() == "t3_ee0obo"

def test_convert_utc_epoch_to_datetime(reddit_psaw):
    """

    """
    ## Test Time (as recorded in the comment above)
    test_time = 1577025930 # 6:45am (Pacific) on Sunday December 22
    ## Convet
    test_time_converted = reddit_psaw.convert_utc_epoch_to_datetime(test_time)
    ## Switch Timezone to Pacific
    utc = pytz.utc
    pacific = pytz.timezone("US/Pacific")
    pacfic_converted = utc.localize(test_time_converted).astimezone(pacific)
    ## Test
    assert pacfic_converted.year == 2019
    assert pacfic_converted.month == 12
    assert pacfic_converted.day == 22
    assert pacfic_converted.hour == 6
    assert pacfic_converted.minute == 45
