
########################
### Imports
########################

## Standard Library
import string

## External Libraries
import pytest

## Local
from smgeo.util.tokenizer import (STOPWORDS,
                                  PRONOUNS,
                                  Tokenizer)

########################
### Globals
########################

## Tokenizer Defaut Initialization Parameters
DEFAULT_TOKENIZER_INIT = dict(
                 stopwords=STOPWORDS,
                 keep_case=False,
                 negate_handling=True,
                 negate_token=False,
                 upper_flag=False,
                 keep_punctuation=False,
                 keep_numbers=False,
                 expand_contractions=True,
                 keep_user_mentions=True,
                 keep_pronouns=False,
                 keep_url=True,
                 keep_hashtags=True,
                 keep_retweets=False,
                 emoji_handling=None,
)

########################
### Tests
########################

def test_filter_stopwords():
    """

    """
    ## Initialize Tokenizer
    tokenizer = Tokenizer(**DEFAULT_TOKENIZER_INIT)
    ## Test Statements
    statements = [
                 "I can\'t wait to go to the movies later. 💔 me some Ashton Kucher!",
                 "You have to be kidding me.",
                 "OH NO! Not the dog!"
                 ]
    expected_output = [
        ["not_wait","go","movies","later", "💔", "ashton", "kucher"],
        ["kidding"],
        ["oh","dog"]
    ]
    ## Check
    for s, e in zip(statements, expected_output):
        s_tokenized = tokenizer.tokenize(s)
        assert s_tokenized == e
    
def test_keep_case():
    """

    """
    ## Initialize Tokenizer
    init_params = DEFAULT_TOKENIZER_INIT.copy()
    init_params["keep_case"] = True
    tokenizer = Tokenizer(**init_params)
    ## Test Statements
    statements = [
                 "I can\'t wait to go to the movies later. 💔 me some Ashton Kucher!",
                 "You have to be kidding me.",
                 "OH NO! Not the dog!"
                 ]
    expected_output = [
        ["not_wait","go","movies","later", "💔", "Ashton", "Kucher"],
        ["kidding"],
        ["OH","dog"]
    ]
    ## Check
    for s, e in zip(statements, expected_output):
        s_tokenized = tokenizer.tokenize(s)
        assert s_tokenized == e

def test_negate_handling():
    """

    """
    ## Test Statements
    statements = [
                 "I do not want to go out tonight",
                 "I can not continue living this way",
                 "I can not not wait until tomorrow!"
                 ]
    ## Initialize Tokenizer Without Handling
    init_params = DEFAULT_TOKENIZER_INIT.copy()
    init_params["stopwords"] = set()
    init_params["negate_handling"] = False
    tokenizer = Tokenizer(**init_params)
    ## Check
    assert [tokenizer.tokenize(i) for i in statements] == [
          ['i', 'do', 'not', 'want', 'to', 'go', 'out', 'tonight'],
          ['i', 'can', 'not', 'continue', 'living', 'this', 'way'],
          ['i', 'can', 'not', 'not', 'wait', 'until', 'tomorrow']]
    ## Initialize Tokenize With Handling
    init_params = DEFAULT_TOKENIZER_INIT.copy()
    init_params["stopwords"] = set()
    init_params["negate_handling"] = True
    tokenizer = Tokenizer(**init_params)    
    assert [tokenizer.tokenize(i) for i in statements] == [
            ['i', 'do', 'not_want', 'to', 'go', 'out', 'tonight'],
            ['i', 'can', 'not_continue', 'living', 'this', 'way'],
            ['i', 'can', 'not_not_wait', 'until', 'tomorrow']]

def test_negate_flag():
    """

    """
    ## Test Statements
    statements = [
                 "I do not want to go out tonight. Not again.",
                 "I can not continue living this way",
                 "I can not not wait until tomorrow!"
                 ]
    ## Initialize Tokenizer
    init_params = DEFAULT_TOKENIZER_INIT.copy()
    init_params["stopwords"] = set()
    init_params["negate_handling"] = True
    init_params["negate_token"] = True
    tokenizer = Tokenizer(**init_params)
    ## Check
    assert [tokenizer.tokenize(i) for i in statements] == [
          ['i', 'do', 'not_want', 'to', 'go', 'out', 'tonight', 'not_again', '<NEGATE_FLAG>', '<NEGATE_FLAG>'],
          ['i', 'can', 'not_continue', 'living', 'this', 'way', '<NEGATE_FLAG>'],
          ['i', 'can', 'not_not_wait', 'until', 'tomorrow', '<NEGATE_FLAG>','<NEGATE_FLAG>']]
    
def test_upper_flag():
    """

    """
    ## Initialize Tokenizer
    init_params = DEFAULT_TOKENIZER_INIT.copy()
    init_params["upper_flag"] = True
    init_params["stopwords"] = set()
    tokenizer = Tokenizer(**init_params)
    ## Test
    test_statement = "HOW CAN YOU NOT KNOW THAT!"
    assert tokenizer.tokenize(test_statement) == \
        ['how', 'can', 'you', 'not_know', 'that', '<UPPER_FLAG>']
    

def test_keep_punctuation():
    """

    """
    ## Initialize Tokenizer
    init_params = DEFAULT_TOKENIZER_INIT.copy()
    init_params["keep_punctuation"] = True
    init_params["stopwords"] = set()
    tokenizer = Tokenizer(**init_params)
    ## Test
    test_statement = "HOW CAN YOU NOT KNOW THAT!?!"
    assert tokenizer.tokenize(test_statement) == \
        ['how', 'can', 'you', 'not_know', 'that', '!?!']
    
def test_keep_numbers():
    """

    """
    ## Initialize Tokenizer
    init_params = DEFAULT_TOKENIZER_INIT.copy()
    init_params["keep_numbers"] = True
    init_params["keep_punctuation"] = False
    init_params["stopwords"] = set()
    tokenizer = Tokenizer(**init_params)
    ## Test
    test_statement = "HOW CAN YOU 2 NOT KNOW THAT in 2019!?!"
    assert tokenizer.tokenize(test_statement) == \
        ['how', 'can', 'you', "<NUMERIC>", 'not_know', 'that', 'in', '<NUMERIC>']
    
def test_expand_contractions():
    """

    """
    ## Initialize Tokenizer
    init_params = DEFAULT_TOKENIZER_INIT.copy()
    init_params["expand_contractions"] = False
    init_params["stopwords"] = set()
    tokenizer = Tokenizer(**init_params)
    ## Test
    test_statement = "I can\'t wait to go to the movies later. Should\'ve gone yesterday."
    assert tokenizer.tokenize(test_statement) == \
        ['i','can', 'not_wait','to','go','to','the','movies','later',"should've",'gone','yesterday']
    ## Initialize Tokenizer Again (Without Negation Handling)
    init_params = DEFAULT_TOKENIZER_INIT.copy()
    init_params["expand_contractions"] = False
    init_params["negate_handling"] = False
    init_params["stopwords"] = set()
    tokenizer = Tokenizer(**init_params)
    assert tokenizer.tokenize(test_statement) == \
        ['i','can\'t', 'wait','to','go','to','the','movies','later',"should've",'gone','yesterday']
    ## Initialize One Last Time (With Expansion)
    init_params = DEFAULT_TOKENIZER_INIT.copy()
    init_params["expand_contractions"] = True
    init_params["negate_handling"] = False
    init_params["stopwords"] = set()
    tokenizer = Tokenizer(**init_params)
    assert tokenizer.tokenize(test_statement) == \
        ['i', 'can', 'not', 'wait', 'to', 'go', 'to', 'the', 'movies', 'later', 'should', 'have', 'gone', 'yesterday']

def test_keep_user_mentions():
    """

    """
    ## Test Statements
    test_statements = ["Going to the movies later with @Friend1.", # Twitter
                       "Calling u/Friend1 to chime in here."] # Reddit
    ## Initialize Tokenizer (Dropping User Mentions)
    init_params = DEFAULT_TOKENIZER_INIT.copy()
    init_params["keep_user_mentions"] = False
    init_params["stopwords"] = set()
    tokenizer = Tokenizer(**init_params)
    assert [tokenizer.tokenize(i) for i in test_statements] == \
                [['going', 'to', 'the', 'movies', 'later', 'with'],
                 ['calling', 'to', 'chime', 'in', 'here']]
    ## Initialize Tokenizer (Keeping User Mentions)
    init_params = DEFAULT_TOKENIZER_INIT.copy()
    init_params["keep_user_mentions"] = True
    init_params["stopwords"] = set()
    tokenizer = Tokenizer(**init_params)
    assert [tokenizer.tokenize(i) for i in test_statements] == \
                [['going', 'to', 'the', 'movies', 'later', 'with', '<USER_MENTION>'],
                 ['calling', '<USER_MENTION>','to', 'chime', 'in', 'here']]
                

def test_keep_pronouns():
    """

    """
    ## Initialize Tokenizer
    init_params = DEFAULT_TOKENIZER_INIT.copy()
    init_params["keep_pronouns"] = True
    tokenizer = Tokenizer(**init_params)
    ## Test Statements
    statements = [
                 "I can\'t wait to go to the movies later. 💔 me some Ashton Kucher!",
                 "You have to be kidding me.",
                 "OH NO! Not the dog!"
                 ]
    expected_output = [
        ["i", "not_wait","go","movies","later", "💔", "me", "ashton", "kucher"],
        ["you","kidding","me"],
        ["oh","dog"]
    ]
    ## Check
    for s, e in zip(statements, expected_output):
        s_tokenized = tokenizer.tokenize(s)
        assert s_tokenized == e

def test_keep_url():
    """

    """
    ## Test Statement
    test_statements = ["Just found a really cool website to help with transportation http://ts.jhu.edu/Shuttles/",
                       "Just found a really cool website to help with transportation ts.jhu.edu/Shuttles/"]
    ## Initialize Tokenizer (Preserving URL)
    init_params = DEFAULT_TOKENIZER_INIT.copy()
    init_params["keep_url"] = True
    tokenizer = Tokenizer(**init_params)
    ## Check
    for t in test_statements:
        assert tokenizer.tokenize(t) == ['found', 'really', 'cool', 'website', 'help', 'transportation', '<URL_TOKEN>']
    ## Initialize Tokenizer (Dropping URL)
    init_params = DEFAULT_TOKENIZER_INIT.copy()
    init_params["keep_url"] = False
    tokenizer = Tokenizer(**init_params)
    ## Check
    for t in test_statements:
        assert tokenizer.tokenize(t) == ['found', 'really', 'cool', 'website', 'help', 'transportation']

def test_keep_hashtag():
    """

    """
    ## Test Statement
    test_statement = "Time to get ready for school #Sucks #IDontWantToLearn"
    ## Initialize Tokenizer (Preserving Hashtags)
    init_params = DEFAULT_TOKENIZER_INIT.copy()
    init_params["keep_hashtags"] = True
    init_params["keep_case"] = True
    tokenizer = Tokenizer(**init_params)
    ## Check
    assert tokenizer.tokenize(test_statement) == \
        ['Time', 'get', 'ready', 'school', 'Sucks', 'IDontWantToLearn']
    ## Initialize Tokenizer (Dropping Hashtags)
    init_params = DEFAULT_TOKENIZER_INIT.copy()
    init_params["keep_hashtags"] = False
    init_params["keep_case"] = True
    tokenizer = Tokenizer(**init_params)
    ## Check
    assert tokenizer.tokenize(test_statement) == \
        ['Time', 'get', 'ready', 'school']

def test_keep_retweets():
    """

    """
    ## Test Statement
    test_statement = "RT: @Friend1 Time to get ready for school #Sucks #IDontWantToLearn"
    ## Initialize Tokenizer (Preserving Retweet)
    init_params = DEFAULT_TOKENIZER_INIT.copy()
    init_params["keep_retweets"] = True
    init_params["keep_case"] = True
    tokenizer = Tokenizer(**init_params)
    assert tokenizer.tokenize(test_statement) == \
        ['<RETWEET>', '<USER_MENTION>', 'Time', 'get', 'ready', 'school', 'Sucks', 'IDontWantToLearn']
    ## Initialize Tokenizer (Preserving Retweet, Dropping Case)
    init_params = DEFAULT_TOKENIZER_INIT.copy()
    init_params["keep_retweets"] = True
    init_params["keep_case"] = False
    tokenizer = Tokenizer(**init_params)
    assert tokenizer.tokenize(test_statement) == \
        ['<RETWEET>', '<USER_MENTION>', 'time', 'get', 'ready', 'school', 'sucks', 'idontwanttolearn']
    ## Initialize Tokenizer (Dropping Retweet)
    init_params = DEFAULT_TOKENIZER_INIT.copy()
    init_params["keep_retweets"] = False
    init_params["keep_case"] = False
    tokenizer = Tokenizer(**init_params)
    assert tokenizer.tokenize(test_statement) == \
        ['<USER_MENTION>', 'time', 'get', 'ready', 'school', 'sucks', 'idontwanttolearn']

def test_emoji_handling():
    """

    """
    ## Sample Text
    text = 'RT @lav09rO5KgJS: Tell em J.T. ! 😂😍http://t.co/Tc_qbFYmFYm'
    ## Test 1 (No Special Handling)
    tokenizer = Tokenizer(**DEFAULT_TOKENIZER_INIT)
    tokens_no_handle = tokenizer.tokenize(text)
    assert tokens_no_handle == ['<USER_MENTION>', 'tell', 'em', 'j.t.', '😂', '😍', '<URL_TOKEN>']
    ## Test 2 (Replace)
    init_params = DEFAULT_TOKENIZER_INIT.copy()
    init_params["emoji_handling"] = "replace"
    tokenizer = Tokenizer(**init_params)
    tokens_replace = tokenizer.tokenize(text)
    assert tokens_replace == ['<USER_MENTION>', 'tell', 'em', 'j.t.', '<EMOJI>', '<EMOJI>', '<URL_TOKEN>']
    ## Test 3 (Strip)
    init_params = DEFAULT_TOKENIZER_INIT.copy()
    init_params["emoji_handling"] = "strip"
    tokenizer = Tokenizer(**init_params)
    tokens_strip = tokenizer.tokenize(text)
    assert tokens_strip == ['<USER_MENTION>', 'tell', 'em', 'j.t.','<URL_TOKEN>']
    ## Test 4 (Error)
    init_params = DEFAULT_TOKENIZER_INIT.copy()
    init_params["emoji_handling"] = "FAKE_ARGUMENT"
    tokenizer = Tokenizer(**init_params)
    with pytest.raises(ValueError):
        _ = tokenizer.tokenize(text)