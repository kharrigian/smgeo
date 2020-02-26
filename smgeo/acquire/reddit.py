
REQUEST_LIMIT = 100000

#####################
### Imports
#####################

## Standard Libary
import os
import json
import datetime

## External Libaries
import pandas as pd
from praw import Reddit
from psaw import PushshiftAPI

#####################
### Wrapper
#####################

class RedditData(object):

    """
    Reddit Data Retrieval via PSAW and PRAW (optionally)
    """

    def __init__(self,
                 init_praw=False):
        """
        Initialize a class to retrieve Reddit data based on
        use case and format into friendly dataframes.

        Args:
            init_praw (bool): If True, retrieves data objects 
                    from Reddit API. Requires existence of 
                    config.json with adequate API credentials
                    in home directory
        
        Returns:
            None
        """
        ## Class Attributes
        self._init_praw = init_praw
        ## Initialize APIs
        self._initialize_api_wrappers()
    
    def __repr__(self):
        """
        Print a description of the class state.

        Args:
            None
        
        Returns:
            desc (str): Class parameters
        """
        desc = "RedditData(init_praw={})".format(self._init_praw)
        return desc

    def _initialize_api_wrappers(self):
        """
        Initialize API Wrappers (PRAW and/or PSAW)

        Args:
            None
        
        Returns:
            None. Sets class api attribute.
        """
        if self._init_praw:
            ## Load Credentials
            config_file = os.path.join(os.path.abspath(__file__).split(__file__)[0],
                                    "config.json")
            config = json.load(open(config_file,"r"))["reddit"]
            self._praw = Reddit(**config)
            ## Initialize API Objects
            self.api = PushshiftAPI(self._praw)
        else:
            ## Initialize API Objects
            self.api = PushshiftAPI()
        
    def _get_start_date(self,
                        start_date_iso=None):
        """
        Get start date epoch

        Args:
            start_date_iso (str or None): If str, expected
                    to be of form "YYYY-MM-DD". If None, 
                    defaults to start of Reddit
        
        Returns:
            start_epoch (int): Start date in form of epoch
        """
        if start_date_iso is None:
            start_date_iso = "2005-08-01"
        start_date_digits = list(map(int,start_date_iso.split("-")))
        start_epoch = int(datetime.datetime(*start_date_digits).timestamp())
        return start_epoch
    
    def _get_end_date(self,
                      end_date_iso=None):
        """
        Get end date epoch

        Args:
            end_date_iso (str or None): If str, expected
                    to be of form "YYYY-MM-DD". If None, 
                    defaults to current date
        
        Returns:
            end_epoch (int): End date in form of epoch
        """
        if end_date_iso is None:
            end_date_iso = (datetime.datetime.now().date() + \
                            datetime.timedelta(1)).isoformat()
        end_date_digits = list(map(int,end_date_iso.split("-")))
        end_epoch = int(datetime.datetime(*end_date_digits).timestamp())
        return end_epoch
    
    def _parse_psaw_submission_request(self,
                                       request):
        """
        Retrieve submission search data and format into 
        a standard pandas dataframe format

        Args:
            request (generator): self.api.search_submissions response
        
        Returns:
            df (pandas DataFrame): Submission search data
        """
        ## Define Variables of Interest
        data_vars = ["archived",
                     "author",
                     "author_flair_text",
                     "author_flair_type",
                     "author_fullname",
                     "category",
                     "comment_limit",
                     "content_categories",
                     "created_utc",
                     "crosspost_parent",
                     "domain",
                     "discussion_type",
                     "distinguished",
                     "downs",
                     "full_link",
                     "gilded",
                     "id",
                     "is_meta",
                     "is_original_content",
                     "is_reddit_media_domain",
                     "is_self",
                     "is_video",
                     "link_flair_text",
                     "link_flair_type",
                     "locked",
                     "media",
                     "num_comments",
                     "num_crossposts",
                     "num_duplicates",
                     "num_reports",
                     "over_18",
                     "permalink",
                     "score",
                     "selftext",
                     "subreddit",
                     "subreddit_id",
                     "thumbnail",
                     "title",
                     "url",
                     "ups",
                     "upvote_ratio"]
        ## Parse Data
        response_formatted = []
        for r in request:
            r_data = {}
            if not self._init_praw:
                for d in data_vars:
                    r_data[d] = None
                    if hasattr(r, d):
                        r_data[d] = getattr(r, d)
            else:
                for d in data_vars:
                    r_data[d] = None
                    if hasattr(r, d):
                        d_obj = getattr(r, d)
                        if d_obj is None:
                            continue
                        if d == "author":
                            d_obj = d_obj.name
                        if d == "created_utc":
                            d_obj = int(d_obj)
                        if d == "subreddit":
                            d_obj = d_obj.display_name
                        r_data[d] = d_obj
            response_formatted.append(r_data)
        ## Format into DataFrame
        df = pd.DataFrame(response_formatted)
        if len(df) > 0:
            df = df.sort_values("created_utc", ascending=True)
            df = df.reset_index(drop=True)
        return df
    
    def _parse_psaw_comment_request(self,
                                    request):
        """
        Retrieve comment search data and format into 
        a standard pandas dataframe format

        Args:
            request (generator): self.api.search_comments response
        
        Returns:
            df (pandas DataFrame): Comment search data
        """
        ## Define Variables of Interest
        data_vars = [
                    "author",
                    "author_flair_text",
                    "author_flair_type",
                    "author_fullname",
                    "body",
                    "collapsed",
                    "collapsed_reason",
                    "controversiality",
                    "created_utc",
                    "downs",
                    "edited",
                    "gildings",
                    "id",
                    "is_submitter",
                    "link_id",
                    "locked",
                    "parent_id",
                    "permalink",
                    "stickied",
                    "subreddit",
                    "subreddit_id",
                    "score",
                    "score_hidden",
                    "total_awards_received",
                    "ups"
        ]
        ## Parse Data
        response_formatted = []
        for r in request:
            r_data = {}
            if not self._init_praw:
                for d in data_vars:
                    r_data[d] = None
                    if hasattr(r, d):
                        r_data[d] = getattr(r, d)
            else:
                for d in data_vars:
                    r_data[d] = None
                    if hasattr(r, d):
                        d_obj = getattr(r, d)
                        if d_obj is None:
                            continue
                        if d == "author":
                            d_obj = d_obj.name
                        if d == "created_utc":
                            d_obj = int(d_obj)
                        if d == "subreddit":
                            d_obj = d_obj.display_name
                        r_data[d] = d_obj
            response_formatted.append(r_data)
        ## Format into DataFrame
        df = pd.DataFrame(response_formatted)
        if len(df) > 0:
            df = df.sort_values("created_utc", ascending=True)
            df = df.reset_index(drop=True)
        return df

    def _getSubComments(self,
                        comment,
                        allComments):
        """

        """
        ## Append Comment
        allComments.append(comment)
        ## Get Replies
        if not hasattr(comment, "replies"):
            replies = comment.comments()
        else:
            replies = comment.replies
        ## Recurse
        for child in replies:
            self._getSubComments(child, allComments)

    def _retrieve_submission_comments_praw(self,
                                           submission_id):
        """

        """
        ## Retrieve Submission
        sub = self._praw.submission(submission_id)
        ## Initialize Comment List
        comments = sub.comments
        ## Recursively Expand Comment Forest
        commentsList = []
        for comment in comments:
            self._getSubComments(comment, commentsList)
        ## Ignore Comment Forest Artifacts
        commentsList = [c for c in commentsList if "MoreComments" not in str(type(c))]
        ## Parse
        if len(commentsList) > 0:
            comment_df = self._parse_psaw_comment_request(commentsList)
            return comment_df

    def retrieve_subreddit_submissions(self,
                                       subreddit,
                                       start_date=None,
                                       end_date=None,
                                       limit=REQUEST_LIMIT):
        """
        Retrieve submissions for a particular subreddit

        Args:
            subreddit (str): Canonical name of the subreddit
            start_date (str or None): If str, expected
                    to be of form "YYYY-MM-DD". If None, 
                    defaults to start of Reddit
            end_date (str or None):  If str, expected
                    to be of form "YYYY-MM-DD". If None, 
                    defaults to current date
            limit (int): Maximum number of submissions to 
                    retrieve
        
        Returns:
            df (pandas dataframe): Submission search data
        """
        ## Get Start/End Epochs
        start_epoch = self._get_start_date(start_date)
        end_epoch = self._get_end_date(end_date)
        ## Construct Call
        req = self.api.search_submissions(after=start_epoch,
                                          before=end_epoch,
                                          subreddit=subreddit,
                                          limit=limit)
        ## Retrieve and Parse Data
        df = self._parse_psaw_submission_request(req)
        if len(df) > 0:
            df = df.sort_values("created_utc", ascending=True)
            df = df.reset_index(drop=True)
        return df
    
    def retrieve_submission_comments(self,
                                     submission):
        """
        Retrieve comments for a particular submission

        Args:
            submission (str): Canonical name of the submission

        Returns:
            df (pandas dataframe): Comment search data
        """
        ## ID Extraction
        if "https" in submission or "reddit" in submission:
            submission = submission.split("comments/")[1].split("/")[0]
        if submission.startswith("t3_"):
            submission = submission.replace("t3_","")
        ## Construct Call
        req = self.api.search_comments(link_id=f"t3_{submission}")
        ## Retrieve and Parse data
        df = self._parse_psaw_comment_request(req)
        ## Fall Back to PRAW
        if self._init_praw and len(df) == 0:
            df = self._retrieve_submission_comments_praw(submission_id=submission)
        if len(df) > 0:
            df = df.sort_values("created_utc", ascending=True)
            df = df.reset_index(drop=True)
        return df
    
    def retrieve_author_comments(self,
                                 author,
                                 start_date=None,
                                 end_date=None,
                                 limit=REQUEST_LIMIT):
        """
        Retrieve comments for a particular Reddit user. Does not
        return user-authored submissions (e.g. self-text)

        Args:
            author (str): Username of the redditor
            start_date (str or None): If str, expected
                    to be of form "YYYY-MM-DD". If None, 
                    defaults to start of Reddit
            end_date (str or None):  If str, expected
                    to be of form "YYYY-MM-DD". If None, 
                    defaults to current date
            limit (int): Maximum number of submissions to 
                    retrieve
        
        Returns:
            df (pandas dataframe): Comment search data
        """
        ## Get Start/End Epochs
        start_epoch = self._get_start_date(start_date)
        end_epoch = self._get_end_date(end_date)
        ## Automatic Limit Detection
        if limit is None:
            if self._init_praw:
                self.api._search_func = self.api._search
            counts = next(self.api.search_comments(author=author,
                                                   before=end_epoch,
                                                   after=start_epoch,
                                                   aggs='author'))
            limit = sum([c["doc_count"] for c in counts["author"]])
            if self._init_praw:
                self.api._search_func = self.api._praw_search
        ## Construct query Params
        query_params = {"before":end_epoch,
                        "after":start_epoch,
                        "limit":limit,
                        "author":author}
        ## Construct Call
        req = self.api.search_comments(**query_params)
        ## Retrieve and Parse Data
        df = self._parse_psaw_comment_request(req)
        if len(df) > 0:
            df = df.sort_values("created_utc", ascending=True)
            df = df.reset_index(drop=True)
        return df
    

    def retrieve_author_submissions(self,
                                    author,
                                    start_date=None,
                                    end_date=None,
                                    limit=REQUEST_LIMIT):
        """
        Retrieve submissions for a particular Reddit user. Does not
        return user-authored comments

        Args:
            author (str): Username of the redditor
            start_date (str or None): If str, expected
                    to be of form "YYYY-MM-DD". If None, 
                    defaults to start of Reddit
            end_date (str or None):  If str, expected
                    to be of form "YYYY-MM-DD". If None, 
                    defaults to current date
            limit (int): Maximum number of submissions to 
                    retrieve
        
        Returns:
            df (pandas dataframe): Comment search data
        """
        ## Get Start/End Epochs
        start_epoch = self._get_start_date(start_date)
        end_epoch = self._get_end_date(end_date)
        ## Automatic Limit Detection
        if limit is None:
            if self._init_praw:
                self.api._search_func = self.api._search
            counts = next(self.api.search_submissions(author=author,
                                                      before=end_epoch,
                                                      after=start_epoch,
                                                      aggs='author'))
            limit = sum([c["doc_count"] for c in counts["author"]])
            if self._init_praw:
                self.api._search_func = self.api._praw_search
        ## Construct query Params
        query_params = {"before":end_epoch,
                        "after":start_epoch,
                        "limit":limit,
                        "author":author}
        ## Construct Call
        req = self.api.search_submissions(**query_params)
        ## Retrieve and Parse Data
        df = self._parse_psaw_submission_request(req)
        if len(df) > 0:
            df = df.sort_values("created_utc", ascending=True)
            df = df.reset_index(drop=True)
        return df

    def search_for_submissions(self,
                               query,
                               subreddit=None,
                               start_date=None,
                               end_date=None,
                               limit=REQUEST_LIMIT):
        """
        Search for submissions based on title

        Args:
            query (str): Title query
            subreddit (str or None): Additional filtering by subreddit.
            start_date (str or None): If str, expected
                    to be of form "YYYY-MM-DD". If None, 
                    defaults to start of Reddit
            end_date (str or None):  If str, expected
                    to be of form "YYYY-MM-DD". If None, 
                    defaults to current date
            limit (int): Maximum number of submissions to 
                    retrieve
        
        Returns:
            df (pandas dataframe): Submission search data
        """
        ## Get Start/End Epochs
        start_epoch = self._get_start_date(start_date)
        end_epoch = self._get_end_date(end_date)
        ## Construct Query
        query_params = {
            "title":'"{}"'.format(query),
            "before":end_epoch,
            "after":start_epoch,
            "limit":limit
        }
        if subreddit is not None:
            query_params["subreddit"] = subreddit
        ## Construct Call
        req = self.api.search_submissions(**query_params)
        ## Retrieve and Parse Data
        df = self._parse_psaw_submission_request(req)
        if len(df) > 0:
            df = df.sort_values("created_utc", ascending=True)
            df = df.reset_index(drop=True)
        return df
    
    def search_for_comments(self,
                            query,
                            subreddit=None,
                            start_date=None,
                            end_date=None,
                            limit=REQUEST_LIMIT):
        """
        Search for comments based on text in body

        Args:
            query (str): Comment query
            subreddit (str or None): Additional filtering by subreddit.
            start_date (str or None): If str, expected
                    to be of form "YYYY-MM-DD". If None, 
                    defaults to start of Reddit
            end_date (str or None):  If str, expected
                    to be of form "YYYY-MM-DD". If None, 
                    defaults to current date
            limit (int): Maximum number of submissions to 
                    retrieve
        
        Returns:
            df (pandas dataframe): Comment search data
        """
        ## Get Start/End Epochs
        start_epoch = self._get_start_date(start_date)
        end_epoch = self._get_end_date(end_date)
        ## Construct Query
        query_params = {
            "q":query,
            "before":end_epoch,
            "after":start_epoch,
            "limit":limit
        }
        if subreddit is not None:
            query_params["subreddit"] = subreddit
        ## Construct Call
        req = self.api.search_comments(**query_params)
        ## Retrieve and Parse Data
        df = self._parse_psaw_comment_request(req)
        if len(df) > 0:
            df = df.sort_values("created_utc", ascending=True)
            df = df.reset_index(drop=True)
        return df
        
    def convert_utc_epoch_to_datetime(self,
                                      epoch):
        """
        Convert an integer epoch time to a datetime

        Args:
            epoch (int): UTC epoch time
        
        Returns:
            conversion (datetime): Datetime object
        """
        ## Convert epoch to datetime
        conversion = datetime.datetime.utcfromtimestamp(epoch)
        return conversion
