
"""
Data loaders for outputs generated from the preprocessing pipeline.
"""

#################
### Imports
#################

## Standard Libary
import json
import gzip
from string import punctuation

## External Libraries
import emoji

## Local Modules
from .tokenizer import (STOPWORDS,
                        PRONOUNS,
                        CONTRACTIONS)
from .helpers import flatten

#################
### Classes
#################

class LoadProcessedData(object):

    """
    Generic Data Loading Class
    """

    def __init__(self,
                 filter_negate=False,
                 filter_upper=False,
                 filter_punctuation=False,
                 filter_numeric=False,
                 filter_user_mentions=False,
                 filter_url=False,
                 filter_retweet=False,
                 filter_stopwords=False,
                 keep_pronouns=True,
                 preserve_case=True,
                 filter_empty=True,
                 emoji_handling=None,
                 max_tokens_per_document=None,
                 max_documents_per_user=None):
        """
        Generic Data Loading Class

        Args:
            filter_negate (bool): Remove <NEGATE_FLAG> tokens
            filter_upper (bool): Remove <UPPER_FLAG> tokens
            filter_punctuation (bool): Remove standalone punctuation
            filter_numeric (bool): Remove <NUMERIC> tokens
            filter_user_mentions (bool): Remove <USER_MENTION> tokens
            filter_url (bool): Remove URL_TOKEN tokens
            filter_retweet (bool): Remove <RETWEET> and proceeding ":" tokens.
            filter_stopwords (bool): Remove stopwords from nltk english stopword set
            keep_pronouns (bool): If removing stopwords, keep pronouns
            preserve_case (bool): Keep token case as is. Otherwise, make lowercase.
            filter_empty (bool): Remove empty strings
            emoji_handling (str or None): If None, emojis are kept as they appear in the text. Otherwise,
                                          should be "replace" or "strip". If "replace", they are replaced
                                          with a generic "<EMOJI>" token. If "strip", they are removed completely.
            max_tokens_per_document (int or None): Only consider the first N tokens from a single document.
                                                   If None (default), will take all tokens.
            max_documents_per_user (int or None): Only consider the most recent N documents from a user. If
                                                  None (default), will take all documents.
        """
        ## Class Arguments
        self.filter_negate = filter_negate
        self.filter_upper = filter_upper
        self.filter_punctuation = filter_punctuation
        self.filter_numeric = filter_numeric
        self.filter_user_mentions = filter_user_mentions
        self.filter_url = filter_url
        self.filter_retweet = filter_retweet
        self.filter_stopwords = filter_stopwords
        self.keep_pronouns = keep_pronouns
        self.preserve_case = preserve_case
        self.filter_empty = filter_empty
        self.emoji_handling = emoji_handling
        self.max_tokens_per_document = max_tokens_per_document
        self.max_documents_per_user = max_documents_per_user
        ## Helpful Variables
        self._punc = set()
        if self.filter_punctuation:
            self._punc = set(punctuation)
        ## Initialization Processes
        self._initialize_filter_set()
        self._initialize_stopwords()

    def _initialize_stopwords(self):
        """
        Initialize stopword set and removes pronouns if desired.

        Args:
            None
        
        Returns:
            None
        """
        ## Format Stopwords into set
        if self.filter_stopwords:
            self.stopwords = set(STOPWORDS)
        else:
            self.stopwords = set()
            return
        ## Contraction Handling
        self.stopwords = self.stopwords | set(self._expand_contractions(list(self.stopwords)))
        ## Pronoun Handling
        if self.keep_pronouns:
            for pro in PRONOUNS:
                if pro in self.stopwords:
                    self.stopwords.remove(pro)

    def _strip_emojis(self,
                      tokens):
        """

        """
        tokens = list(filter(lambda t: t not in emoji.UNICODE_EMOJI and t != "<EMOJI>", tokens))
        return tokens
    
    def _replace_emojis(self,
                        tokens):
        """

        """
        tokens = list(map(lambda t: "<EMOJI>" if t in emoji.UNICODE_EMOJI else t, tokens))
        return tokens

    def _expand_contractions(self,
                             tokens):
        """
        Expand English contractions.

        Args:
            tokens (list of str): Token list
        
        Returns:
            tokens (list of str): Tokens, now with expanded contractions.
        """
        tokens = \
        flatten(list(map(lambda t: CONTRACTIONS[t.lower()].split() if t.lower() in CONTRACTIONS else [t],
                         tokens)))
        return tokens
    
    def _select_n_recent_documents(self,
                                   user_data):
        """
        Select the N most recent documents in user data based on the 
        class initialization parameters.

        Args:
            user_data (list of dict): Processed user data dictionaries
        
        Returns:
            user_data (list of dict): N most recent documents
        """
        ## Downsample Documents
        if self.max_documents_per_user is not None:
            user_data = sorted(user_data, key = lambda x: x["created_utc"], reverse = True)
            user_data = user_data[:min(len(user_data), self.max_documents_per_user)]
        return user_data
    
    def load_user_data(self,
                       filename):
        """
        Load preprocessed user data from disk.

        Args:
            filename (str): Path to .tar.gz file
        
        Returns:
            user_data (list of dict): Preprocessed, filtered user
                                      data
        """
        ## Load the GZIPed file
        with gzip.open(filename) as the_file:
            user_data = json.load(the_file)
        ## Data Amount Filtering
        user_data = self._select_n_recent_documents(user_data)
        ## Apply Processing
        user_data = self.filter_user_data(user_data)
        return user_data
    
    def load_user_metadata(self,
                           filename):
        """
        Load user metedata file

        Args:
            filename (str): Path to .tar.gz file
        
        Returns:
            user_data (dict): User metadata dictionary
        """
        ## Load the GZIPed file
        with gzip.open(filename) as the_file:
            user_data = json.load(the_file)
        return user_data
    
    def _filter_in(self,
                   obj,
                   ignore_set):
        """
        Filter a list by excluding matches in a set

        Args:
            obj (list): List to be filtered
            ignore_set (iterable): Set to check items again
        
        Returns:
            filtered_obj (list): Original list excluding objects
                          found in the ignore_set
        """
        return list(filter(lambda l: l not in ignore_set, obj))
    
    def _initialize_filter_set(self):
        """
        Initialize the set of items to filter from
        a tokenized text list based on class initialization
        parameters.

        Args:
            None
        
        Returns:
            None, assigns self.filter_set attribute
        """
        ## Initialize SEt
        self.filter_set = set()
        if self.filter_negate:
            self.filter_set.add("<NEGATE_FLAG>")
        ## Filter Upper
        if self.filter_upper:
             self.filter_set.add("<UPPER_FLAG>")
        ## Filter Numeric
        if self.filter_numeric:
           self.filter_set.add("<NUMERIC>")
        ## Filter User Mentions
        if self.filter_user_mentions:
            self.filter_set.add("<USER_MENTION>")
        ## Filter URL
        if self.filter_url:
            self.filter_set.add("<URL_TOKEN>")
        ## Filter Empty Strings
        if self.filter_empty:
            self.filter_set.add("''")
            self.filter_set.add('""')
    
    def filter_user_data(self,
                         user_data):
        """
        Filter loaded user data based on class initialization
        parameters.

        Args:
            user_data (list of dict): Preprocessed user data
        
        Returns:
            filtered_data (list of dict): Filtered user data
        """
        ## Tokenized Text Field
        tt = "text_tokenized"
        ## Initialize Filtered Data Cache
        filtered_data = []
        for i, d in enumerate(user_data):
            ## Filter Based on Ignore Set
            d[tt] = self._filter_in(d[tt], self.filter_set)
            ## Length Check
            if len(d[tt]) == 0:
                filtered_data.append(d)
                continue
            ## Filter Retweets
            if self.filter_retweet and d[tt][0] == "<RETWEET>":
                d[tt] = d[tt][1:]
                for _ in range(2):
                    if len(d[tt]) == 0:
                        break
                    if d[tt][0] in ["<USER_MENTION>", ":"]:
                        d[tt] = d[tt][1:]
            ## Max Tokens
            if self.max_tokens_per_document is not None:
                d[tt] = d[tt][:min(len(d[tt]), self.max_tokens_per_document)]
            ## Filter Stopwords
            if self.filter_stopwords:
                d[tt] = list(filter(lambda x: x.lower().replace("not_","") not in self.stopwords, d[tt]))
            ## Filter Punctuation
            if self.filter_punctuation:
                d[tt] = list(filter(lambda i: not all(char in self._punc for char in i), d[tt]))
            ## Case Formatting
            if not self.preserve_case:
                d[tt] = list(map(lambda tok: tok.lower() if tok not in self.filter_set else tok, d[tt]))
            ## Emoji Handling
            if self.emoji_handling is not None:
                if self.emoji_handling == "replace":
                    d[tt] = self._replace_emojis(d[tt])
                elif self.emoji_handling == "strip":
                    d[tt] = self._strip_emojis(d[tt])
                else:
                    raise ValueError("emoji_handling should be 'replace', 'strip', or None.")
            filtered_data.append(d)
        return filtered_data


