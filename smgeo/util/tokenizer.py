
"""
General text tokenizer, geared toward social media text.
"""

####################
### Imports
####################

## Standard Library
import string
import re
import sys
from html.parser import HTMLParser

## External Libraries
import emoji

####################
### Resources
####################

## Stopwords (These are English stopwords from NLTK)
STOPWORDS = set(['i',
                'me',
                'my',
                'myself',
                'we',
                'our',
                'ours',
                'ourselves',
                'you',
                "you're",
                "you've",
                "you'll",
                "you'd",
                'your',
                'yours',
                'yourself',
                'yourselves',
                'he',
                'him',
                'his',
                'himself',
                'she',
                "she's",
                'her',
                'hers',
                'herself',
                'it',
                "it's",
                'its',
                'itself',
                'they',
                'them',
                'their',
                'theirs',
                'themselves',
                'what',
                'which',
                'who',
                'whom',
                'this',
                'that',
                "that'll",
                'these',
                'those',
                'am',
                'is',
                'are',
                'was',
                'were',
                'be',
                'been',
                'being',
                'have',
                'has',
                'had',
                'having',
                'do',
                'does',
                'did',
                'doing',
                'a',
                'an',
                'the',
                'and',
                'but',
                'if',
                'or',
                'because',
                'as',
                'until',
                'while',
                'of',
                'at',
                'by',
                'for',
                'with',
                'about',
                'against',
                'between',
                'into',
                'through',
                'during',
                'before',
                'after',
                'above',
                'below',
                'to',
                'from',
                'up',
                'down',
                'in',
                'out',
                'on',
                'off',
                'over',
                'under',
                'again',
                'further',
                'then',
                'once',
                'here',
                'there',
                'when',
                'where',
                'why',
                'how',
                'all',
                'any',
                'both',
                'each',
                'few',
                'more',
                'most',
                'other',
                'some',
                'such',
                'no',
                'nor',
                'not',
                'only',
                'own',
                'same',
                'so',
                'than',
                'too',
                'very',
                's',
                't',
                'can',
                'will',
                'just',
                'don',
                "don't",
                'should',
                "should've",
                'now',
                'd',
                'll',
                'm',
                'o',
                're',
                've',
                'y',
                'ain',
                'aren',
                "aren't",
                'couldn',
                "couldn't",
                'didn',
                "didn't",
                'doesn',
                "doesn't",
                'hadn',
                "hadn't",
                'hasn',
                "hasn't",
                'haven',
                "haven't",
                'isn',
                "isn't",
                'ma',
                'mightn',
                "mightn't",
                'mustn',
                "mustn't",
                'needn',
                "needn't",
                'shan',
                "shan't",
                'shouldn',
                "shouldn't",
                'wasn',
                "wasn't",
                'weren',
                "weren't",
                'won',
                "won't",
                'wouldn',
                "wouldn't"])

## Pronouns
PRONOUNS = set([ 
            "he",
            "she",
            "they",
            "i",
            "him",
            "her",
            "we",
            "me",
            "it",
            "you",
            "us",
            "them",
            "myself",
            "ourselves",
            "yourself",
            "yourselves",
            "himself",
            "itself",
            "herself",
            "themselves",
            "my",
            "our",
            "ours",
            "your",
            "yours",
            "their",
            "its",
            "mine",
            "theirs"
])

CONTRACTIONS =  { 
            "ain't": "is not",
            "aren't": "are not",
            "can't": "can not",
            "can't've": "can not have",
            "cannot": "can not",
            "'cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "couldn't've": "could not have",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hadn't've": "had not have",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'd've": "he would have",
            "he'll": "he will",
            "he'll've": "he will have",
            "he's": "he is",
            "how'd": "how did",
            "how'd'y": "how do you",
            "how'll": "how will",
            "how's": "how is",
            "i'd": "i would",
            "i'd've": "i would have",
            "i'll": "i will",
            "i'll've": "i will have",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd": "it would",
            "it'd've": "it would have",
            "it'll": "it will",
            "it'll've": "it will have",
            "it's": "it is",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "mightn't've": "might not have",
            "must've": "must have",
            "mustn't": "must not",
            "mustn't've": "must not have",
            "needn't": "need not",
            "needn't've": "need not have",
            "o'clock": "of the clock",
            "oughtn't": "ought not",
            "oughtn't've": "ought not have",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "shan't've": "shall not have",
            "she'd": "she would",
            "she'd've": "she would have",
            "she'll": "she will",
            "she'll've": "she will have",
            "she's": "she is",
            "should've": "should have",
            "shouldn't": "should not",
            "shouldn't've": "should not have",
            "so've": "so have",
            "so's": "so is",
            "that'd": "that would",
            "that'd've": "that would have",
            "that's": "that is",
            "there'd": "there would",
            "there'd've": "there would have",
            "there's": "there is",
            "they'd": "they would",
            "they'd've": "they would have",
            "they'll": "they will",
            "they'll've": "they will have",
            "they're": "they are",
            "they've": "they have",
            "to've": "to have",
            "wasn't": "was not",
            "we'd": "we would",
            "we'd've": "we would have",
            "we'll": "we will",
            "we'll've": "we will have",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what'll've": "what will have",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "when's": "when is",
            "when've": "when have",
            "where'd": "where did",
            "where's": "where is",
            "where've": "where have",
            "who'll": "who will",
            "who'll've": "who will have",
            "who's": "who is",
            "who've": "who have",
            "why's": "why is",
            "why've": "why have",
            "will've": "will have",
            "won't": "will not",
            "won't've": "will not have",
            "would've": "would have",
            "wouldn't": "would not",
            "wouldn't've": "would not have",
            "y'all": "you all",
            "y'all'd": "you all would",
            "y'all'd've": "you all would have",
            "y'all're": "you all are",
            "y'all've": "you all have",
            "you'd": "you would",
            "you'd've": "you would have",
            "you'll": "you will",
            "you'll've": "you will have",
            "you're": "you are",
            "you've": "you have",
            "that'll": "that will",
            }

PUNCTUATION = string.punctuation
PUNCTUATION += "“”…‘’´"

####################
### Regular Expressions (modified copy of twokenize.py for Python 3 Only)
####################

def regex_or(*items):
    """
    Format regular expression (or statement)

    Args:
        items (regex): Regular Expression
    
    Returns:
        capturing regex
    """
    return '(?:' + '|'.join(items) + ')'

## Whitespace
Whitespace = re.compile("[\s\u0020\u00a0\u1680\u180e\u202f\u205f\u3000\u2000-\u200a]+", re.UNICODE)

## Numeric Combinations
numComb	= "[\u0024\u058f\u060b\u09f2\u09f3\u09fb\u0af1\u0bf9\u0e3f\u17db\ua838\ufdfc\ufe69\uff04\uffe0\uffe1\uffe5\uffe6\u00a2-\u00a5\u20a0-\u20b9]?\\d+(?:\\.\\d+)+%?"

## Punctuation
punctChars = r"['\"“”‘’.?!…,:;]"
punctSeq = r"['\"“”‘’]+|[.?!,…]+|[:;]+"
entity = r"&(?:amp|lt|gt|quot);"

## URLs
urlStart1 = r"(?:https?://|\bwww\.)"
commonTLDs = r"(?:com|org|edu|gov|net|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|pro|tel|travel|xxx)"
ccTLDs = r"(?:ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|" + \
r"bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|" + \
r"er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|" + \
r"hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|" + \
r"lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|" + \
r"nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|sk|" + \
r"sl|sm|sn|so|sr|ss|st|su|sv|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|" + \
r"va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|za|zm|zw)"
urlStart2 = r"\b(?:[A-Za-z\d-])+(?:\.[A-Za-z0-9]+){0,3}\." + regex_or(commonTLDs, ccTLDs) + r"(?:\."+ccTLDs+r")?(?=\W|$)"
urlBody = r"(?:[^\.\s<>][^\s<>]*?)?"
urlExtraCrapBeforeEnd = regex_or(punctChars, entity) + "+?"
urlEnd = r"(?:\.\.+|[<>]|\s|$)"
url = regex_or(urlStart1, urlStart2) + urlBody + "(?=(?:"+urlExtraCrapBeforeEnd+")?"+urlEnd+")"

## Numeric
timeLike = r"\d+(?::\d+){1,2}"
numberWithCommas = r"(?:(?<!\d)\d{1,3},)+?\d{3}" + r"(?=(?:[^,\d]|$))"

## Abbreviations
boundaryNotDot = regex_or("$", r"\s", r"[“\"?!,:;]", entity)
aa1 = r"(?:[A-Za-z]\.){2,}(?=" + boundaryNotDot + ")"
aa2 = r"[^A-Za-z](?:[A-Za-z]\.){1,}[A-Za-z](?=" + boundaryNotDot + ")"
standardAbbreviations = r"\b(?:[Mm]r|[Mm]rs|[Mm]s|[Dd]r|[Ss]r|[Jj]r|[Rr]ep|[Ss]en|[Ss]t)\."
arbitraryAbbrev = regex_or(aa1, aa2, standardAbbreviations)
separators = "(?:--+|―|—|~|–|=)"

## Decorations
decorations = "(?:[♫♪]+|[★☆]+|[♥❤♡]+|[\u2639-\u263b]+|[\ue001-\uebbb]+)"
Hearts = "(?:<+/?3+)+"  # the other hearts are in decorations

## Emoticons
normalEyes = "[:=]" # 8 and x are eyes but cause problems
wink = "[;]"
noseArea = "(?:|-|[^a-zA-Z0-9 ])" # doesn't get :'-(
happyMouths = r"[D\)\]\}]+"
sadMouths = r"[\(\[\{]+"
tongue = "[pPd3]+"
otherMouths = r"(?:[oO]+|[/\\]+|[vV]+|[Ss]+|[|]+)" # remove forward slash if http://'s aren't cleaned
bfLeft = "(♥|0|[oO]|°|[vV]|\\$|[tT]|[xX]|;|\u0ca0|@|ʘ|•|・|◕|\\^|¬|\\*)"
bfCenter = r"(?:[\.]|[_-]+)"
bfRight = r"\2"
s3 = r"(?:--['\"])"
s4 = r"(?:<|&lt;|>|&gt;)[\._-]+(?:<|&lt;|>|&gt;)"
s5 = "(?:[.][_]+[.])"
basicface = "(?:" + bfLeft + bfCenter + bfRight + ")|" + s3 + "|" + s4 + "|" + s5
eeLeft = r"[＼\\ƪԄ\(（<>;ヽ\-=~\*]+"
eeRight = "[\\-=\\);'\u0022<>ʃ）/／ノﾉ丿╯σっµ~\\*]+"
eeSymbol = r"[^A-Za-z0-9\s\(\)\*:=-]"
eastEmote = eeLeft + "(?:"+basicface+"|" +eeSymbol+")+" + eeRight
oOEmote = r"(?:[oO]" + bfCenter + r"[oO])"
emoticon = regex_or(
        "(?:>|&gt;)?" + regex_or(normalEyes, wink) + regex_or(noseArea, "[Oo]")
        + regex_or(tongue+r"(?=\W|$|RT|rt|Rt)", otherMouths + r"(?=\W|$|RT|rt|Rt)", sadMouths, happyMouths),
        regex_or("(?<=(?: ))", "(?<=(?:^))") + regex_or(sadMouths, happyMouths, otherMouths)
        + noseArea + regex_or(normalEyes, wink) + "(?:<|&lt;)?",
        eastEmote.replace("2", "1", 1), basicface,
        oOEmote
)

## Emojis
if "en" in emoji.UNICODE_EMOJI.keys():
    EMOJI_DICT = emoji.UNICODE_EMOJI["en"]
else:
    EMOJI_DICT = emoji.UNICODE_EMOJI
emojis_list = map(lambda x: ''.join(x.split()), EMOJI_DICT.keys())
emoji_r = re.compile('|'.join(re.escape(p) for p in emojis_list))

## Hashtags and at mentions
Hashtag = "#[a-zA-Z0-9_]+" 
AtMention = "[@＠][a-zA-Z0-9_]+"

## Emails
Bound = r"(?:\W|^|$)"
Email = regex_or("(?<=(?:\W))", "(?<=(?:^))") + r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}(?=" +Bound+")"

## Splitters
thingsThatSplitWords = r"[^\s\.,?\"]"
embeddedApostrophe = thingsThatSplitWords+r"+['’′]" + thingsThatSplitWords + "*"

## Arrows
Arrows = regex_or(r"(?:<*[-―—=]*>+|<+[-―—=]*>*)", "[\u2190-\u21ff]+")

## Protected Expressions
Protected = re.compile(
    regex_or(
        Hearts,
        url,
        Email,
        timeLike,
        numberWithCommas,
        numComb,
        emoticon,
        Arrows,
        entity,
        punctSeq,
        arbitraryAbbrev,
        separators,
        decorations,
        embeddedApostrophe,
        Hashtag,
        AtMention), re.UNICODE)

## Edge Punctuation
edgePunctChars = "'\"“”‘’«»{}\\(\\)\\[\\]\\*&" #add \\p{So}? (symbols)
edgePunct = "[" + edgePunctChars + "]"
notEdgePunct = "[a-zA-Z0-9]" # content characters
offEdge = r"(^|$|:|;|\s|\.|,)"  # colon here gets "(hello):" ==> "( hello ):"
EdgePunctLeft = re.compile(offEdge + "("+edgePunct+"+)("+notEdgePunct+")", re.UNICODE)
EdgePunctRight = re.compile("("+notEdgePunct+")("+edgePunct+"+)" + offEdge, re.UNICODE)

## Split the punctuation along the edge
def splitEdgePunct(my_input):
    """

    """
    my_input = EdgePunctLeft.sub(r"\1\2 \3", my_input)
    my_input = EdgePunctRight.sub(r"\1 \2\3", my_input)
    return my_input

# The main work of tokenizing a tweet.
def simpleTokenize(text):
    """

    """
    ## Process the Edges
    splitPunctText = splitEdgePunct(text)
    textLength = len(splitPunctText)
    ## Find Matches for Subsquences that should be Protected
    bads = []
    badSpans = []
    for match in Protected.finditer(splitPunctText):
        if (match.start() != match.end()):
            bads.append( [splitPunctText[match.start():match.end()]] )
            badSpans.append( (match.start(), match.end()) )
    ## Create List of Good Indices for Splitting
    indices = [0]
    for (first, second) in badSpans:
        indices.append(first)
        indices.append(second)
    indices.append(textLength)
    ## Group the indices and map them to their respective portion of the string
    splitGoods = []
    for i in range(0, len(indices), 2):
        goodstr = splitPunctText[indices[i]:indices[i+1]]
        splitstr = goodstr.strip().split(" ")
        splitGoods.append(splitstr)
    ## Reinterpolate Lists
    zippedStr = []
    for i in range(len(bads)):
        zippedStr = addAllnonempty(zippedStr, splitGoods[i])
        zippedStr = addAllnonempty(zippedStr, bads[i])
    zippedStr = addAllnonempty(zippedStr, splitGoods[len(bads)])

    return zippedStr

def addAllnonempty(master,
                   smaller):
    """
    Retrieve Non-empty Groups
    """
    for s in smaller:
        strim = s.strip()
        if (len(strim) > 0):
            master.append(strim)
    return master

## Squeeze Whitespace
def squeezeWhitespace(input):
    """
    "foo   bar " => "foo bar"
    """
    return Whitespace.sub(" ", input).strip()

## Text Normalization
def normalize_text(text):
    """
    Twitter text comes HTML-escaped, so unescape it.
    We also first unescape &amp;'s, in case the text has been buggily double-escaped.
    """
    text = text.replace("&amp;", "&")
    text = HTMLParser().unescape(text)
    return text

####################
### Helpers
####################

def get_ngrams(tokens,
               min_n=1,
               max_n=1):
    """
    Get n-gram tuples from a list of tokens

    Args:
        tokens (list): List of strings
        min_n (int): Minimum n-gram
        max_n (int): Maximum ngram
    
    Returns:
        all_ngrams (list of tuples): Ngram lists.
    """
    ## Check Inputs
    if min_n > max_n:
        raise ValueError("min_n must be less than max_n")
    if min_n == 0:
        raise ValueError("min_n should be greater than 0")
    ## Generate N-Gram Tuples
    all_ngrams = []
    for n in range(min_n, max_n+1):
        all_ngrams.extend(list(zip(*[tokens[i:] for i in range(n)])))
    return all_ngrams

## Emoji Separation
def split_emojis(t):
    """
    Separate a span of emojis into multiple separate
    emojis

    Args:
        t (str): Input string that potentially has emojis
    
    Returns:
        split_t (list): List of emojis split separately
    """
    emojis_found = emoji_r.findall(t)
    split_t = []
    cur_ind = 0
    for matched in emoji_r.finditer(t):
        start, stop = matched.span()
        if cur_ind != start:
            split_t.append(t[cur_ind:start])
        split_t.append(t[start:stop])
        cur_ind = stop
    if cur_ind < len(t) - 1:
        split_t.append(t[cur_ind:])
    return split_t

## Flattening List of Lists
def flatten(l):
    """
    Flatten a list of lists by one level.

    Args:
        l (list of lists): List of lists

    Returns:
        flattened_list (list): Flattened list
    """
    flattened_list = [item for sublist in l for item in sublist]
    return flattened_list
    
####################
### Tokenizer
####################

class Tokenizer(object):

    """
    Tokenizer. Can be imported by itself if desired.
    """

    def __init__(self,
                 stopwords=STOPWORDS,
                 keep_case=False,
                 negate_handling=True,
                 negate_token=False,
                 upper_flag=False,
                 keep_punctuation=False,
                 keep_numbers=False,
                 expand_contractions=True,
                 keep_user_mentions=True,
                 keep_pronouns=True,
                 keep_url=True,
                 keep_hashtags=True,
                 keep_retweets=False,
                 emoji_handling=None,
                 strip_hashtag=True):
        """
        Text Tokenizer. Uses twokenizer to do text splitting. Applies various 
        forms of post-processing.

        Args:
            stopwords (list or None): Stopword list. Default is English stopwords from NLTK.
            keep_case (bool): If False, make tokens lowercase
            negate_handling (bool): If True, modify proceeding token. For
                                example, "Can not wait" -> "Can not_wait"
            negate_token (bool): If negate_handling, add a <NEGATE> token for every
                        negation used in a line of text
            upper_flag (bool): If entire piece of text is uppercase, add <UPPER_FLAG> token.
            keep_punctuation (bool): If False, remove any standard token-splitting punctuation.
            keep_numbers (bool): If False, remove numbers altogether. Otherwise,
                            replace numbers with a <NUMERIC> token.
            expand_contractions (bool): If True, expand contractions (lowercase form).
            keep_user_mentions (bool): If False, remove user mentions altogether. Otherwise,
                            replace them with a generic <USER_MENTION> token.
            keep_pronouns (bool): If True, remove any pronouns from the stopword set.
            keep_url (bool): If True, replace URLs with generic <URL_TOKEN> token. Otherwise,
                        remove urls entirely
            keep_hashtags (bool): If True, strip "#" from hashtag. Otherwise, remove token 
                                altogether.
            keep_retweets (bool): If True, replace "RT" token with "<RETWEET>". Otherwise,
                            remove the token altogether.
            emoji_handling (str or None): If None, emojis are kept as they appear in the text. Otherwise,
                                          should be "replace" or "strip". If "replace", they are replaced
                                          with a generic "<EMOJI>" token. If "strip", they are removed completely.
            strip_hashtag (bool): If True (default), the hashtag at the start of an identified hashtag
                                  is removed. Otherwise, it is replaced with a "HASHTAG=" prefix
        """
        ## Class Attributes
        self.stopwords = stopwords
        self.keep_case = keep_case
        self.negate_token = negate_token
        self.negate_handling = negate_handling
        self.upper_flag = upper_flag
        self.keep_punctuation = keep_punctuation
        self.keep_numbers = keep_numbers
        self.expand_contractions = expand_contractions
        self.keep_user_mentions = keep_user_mentions
        self.keep_pronouns = keep_pronouns
        self.keep_url = keep_url
        self.keep_hashtags = keep_hashtags
        self.keep_retweets = keep_retweets
        self.emoji_handling = emoji_handling
        self.strip_hashtag = strip_hashtag
        ## Class Initialization Procedures
        self._initialize_stopwords()
    
    def __repr__(self):
        """
        Human-readable description of the class

        Args:
            None
        
        Returns:
            desc (str): Summary of class attributes
        """
        fmt_str = []
        for attr in ["keep_case",
                     "negate_handling",
                     "negate_token",
                     "upper_flag",
                     "keep_punctuation",
                     "keep_numbers",
                     "expand_contractions",
                     "keep_user_mentions",
                     "keep_pronouns",
                     "keep_url",
                     "keep_hashtags",
                     "keep_retweets",
                     "emoji_handling"]:
            if hasattr(self, attr):
                fmt_str.append(f"{attr}={getattr(self,attr)}")
        fmt_str = ", ".join(fmt_str)
        desc = f"Tokenizer(stopwords={self.stopwords is not None}, {fmt_str})"
        return desc
    
    def _initialize_stopwords(self):
        """
        Initialize stopword set and removes pronouns if desired.

        Args:
            None
        
        Returns:
            None
        """
        ## Format Stopwords into set if None
        if hasattr(self, "stopwords") and self.stopwords is not None:
            self.stopwords = set(self.stopwords)
        else:
            self.stopwords = set()
        ## Contraction Handling
        if hasattr(self, "expand_contractions") and self.expand_contractions:
            self.stopwords = set(self._expand_contractions(list(self.stopwords)))
        ## Pronoun Handling
        if hasattr(self, "keep_pronouns") and self.keep_pronouns:
            for s in list(self.stopwords):
                if s in PRONOUNS:
                    self.stopwords.remove(s)

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
    
    def _strip_user_mentions(self,
                             tokens):
        """
        Remove tokens mentioning a username (Reddit or Twitter).

        Args:
            tokens (list of str): Token list
        
        Returns:
            tokens (list of str): Tokens, now without usernames.
        """
        tokens = list(filter(lambda t: not (t.startswith("u/") or t.startswith("@")), tokens))
        return tokens
    
    def _replace_user_mentions(self,
                               tokens):
        """
        Replace mention of Reddit and Twitter usernames with "<USER_MENTION>".

        Args:
            tokens (list of str): Token list
        
        Returns:
            tokens (list of str): Tokens, now with generic token in place of usernames.
        """
        tokens = list(map(lambda t: "<USER_MENTION>" if (t.startswith("u/") or t.startswith("@")) else t, tokens))
        return tokens
    
    def _strip_retweets(self,
                        tokens):
        """
        Remove retweet token from the token list.

        Args:
            tokens (list of str): Token list
        
        Returns:
            tokens (list of str): Tokens without RT token
        """
        tokens = list(filter(lambda t: t.lower() != "rt", tokens))
        if len(tokens) > 1 and tokens[0] == ":":
            tokens = tokens[1:]
        return tokens
    
    def _replace_retweets(self,
                          tokens):
        """
        Replace retween token with explicit <RETWEET> token.

        Args:
            tokens (list of str): Token list
        
        Returns:
            tokens (list of str): Tokens with explicit <RETWEET> token if applicable.
        """
        tokens = list(map(lambda t: "<RETWEET>" if t.lower() == "rt" else t, tokens))
        return tokens
    
    def _upper_flag(self,
                    tokens):
        """
        Add <UPPER_FLAG> token to token list of everything is uppercase.

        Args:
            tokens (list of str): Token list
        
        Returns:
            tokens (list of str): Tokens, now with an added <UPPER_FLAG> token if applicable.
        """
        if all(t.isupper() for t in tokens if not all(char in PUNCTUATION for char in t) and not any(char.isdigit() for char in t)):
            tokens.append("<UPPER_FLAG>")
        return tokens
    
    def _strip_punctuation(self,
                           tokens):
        """
        Remove standalone punctuation from the token list

        Args:
            tokens (list of str): Token list
        
        Returns:
            tokens (list of str): Tokens, now without standalone punctuation.
        """
        tokens = list(filter(lambda t: not all(char in PUNCTUATION for char in t), tokens))
        return tokens
    
    def _remove_hashtag(self,
                       tokens):
        """
        Remove tokens that start with hashtags

        Args:
            tokens (list of str): Token list
        
        Returns:
            tokens (list of str): Tokens, now without hashtags.
        """
        tokens = list(filter(lambda t: not t.startswith("#"), tokens))
        return tokens
    
    def _clean_hashtag(self,
                       tokens):
        """
        Strip hashtags of their "#" symbol, but keep the main token if self.strip_hashtag
        is set true. Otherwise, the "#" symbol is replaced with "<HASHTAG=*>" for token *

        Args:
            tokens (list of str): Token list
        
        Returns:
            tokens (list of str): Tokens, now without preceeding "#".
        """
        if not hasattr(self, "strip_hashtag") or self.strip_hashtag:
            tokens = list(map(lambda t: t.lstrip("#") if t.startswith("#") and len(t) > 1 else t, tokens))
        else:
            tokens = list(map(lambda t: "<HASHTAG={}>".format(t[1:]) if t.startswith("#") and len(t) > 1 else t, tokens))
        return tokens
    
    def _strip_url(self,
                   tokens):
        """
        Remove any tokens that are matched as URLs

        Args:
            tokens (list of str): Token list
        
        Returns:
            tokens (list of str): Tokens, now without URLs.
        """
        tokens = list(filter(lambda t: re.match(url, t) is None, tokens))
        return tokens
    
    def _replace_url(self,
                     tokens):
        """
        Replace URLs with generic <URL_TOKEN>.

        Args:
            tokens (list of str): Token list
        
        Returns:
            tokens (list of str): Tokens, now with generic <URL_TOKEN> where applicable.
        """
        tokens = list(map(lambda t: "<URL_TOKEN>" if re.match(url, t) is not None else t, tokens))
        return tokens
    
    def _handle_negations(self,
                          tokens):
        """
        Add not_ to proceeding token when encountered and remove
        the actual "not" mention. If desired, add <NEGATE_FLAG> token
        for each negation.

        Args:
            tokens (list of str): Token list
        
        Returns:
            tokens (list of str): Tokens, now with improved negation handling and 
                        negation flag if desired.
        """
        negated_tokens = []
        NEGATE_FLAG = False
        negations = 0
        negate_prefix = ""
        for i, t in enumerate(tokens):
            if NEGATE_FLAG and t.lower() != "not":
                t = f"{negate_prefix}{t}"
                NEGATE_FLAG = False
                negate_prefix = ""
                negated_tokens.append([t])
                continue
            if t.lower() == "not":
                NEGATE_FLAG = True
                negate_prefix = negate_prefix + "not_"
                negations += 1
                continue
            if t.lower() in CONTRACTIONS and "not" in CONTRACTIONS[t.lower()]:
                t = CONTRACTIONS[t.lower()].split()
                negate_prefix = negate_prefix + "not_"
                negated_tokens.append([t[0]])
                NEGATE_FLAG = True
                negations += 1
                continue
            negated_tokens.append([t])
        negated_tokens = flatten(negated_tokens)
        if hasattr(self, "negate_token") and self.negate_token:
            for _ in range(negations):
                negated_tokens.append("<NEGATE_FLAG>")
        return negated_tokens
    
    def _strip_numbers(self,
                       tokens):
        """
        Remove tokens that contain numbers from the token set.

        Args:
            tokens (list of str): Token list
        
        Returns:
            tokens (list of str): Tokens, now excluding tokens that have digits.
        """
        tokens = list(filter(lambda t:  not any(char.isdigit() for char in t) or \
                                        t.startswith("#") or \
                                        t.startswith("@") or \
                                        t.startswith("u/"),
                                        tokens))
        return tokens

    def _replace_numbers(self,
                         tokens):
        """
        Replace numbers in the token set with a generic <NUMERIC> token.

        Args:
            tokens (list of str): Token list
        
        Returns:
            tokens (list of str): Tokens, now with <NUMERIC> in place of tokens
                        that contain digits.
        """
        tokens = list(map(lambda t: "<NUMERIC>" if (any(char.isdigit() for char in t) and \
                                                    not t.startswith("#") and \
                                                    not t.startswith("@") and \
                                                    not t.startswith("u/")) else t, tokens))
        return tokens
    
    def _expand_emoji_groups(self,
                             tokens):
        """
        Split a series of multiple emojis

        Args:
            tokens (list): List of token strings
        
        Returns:
            tokens (list): List of tokens, with emoji groups split separately
        """
        tokens = list(map(lambda t: split_emojis(t) if any(char in EMOJI_DICT for char in t) else [t], tokens))
        tokens = flatten(tokens)
        return tokens
    
    def _strip_emojis(self,
                      tokens):
        """
        Remove emojis from a list of tokens

        Args:
            tokens (list): Input list of strings
        
        Returns:
            tokens (list): Token list without any emojis
        """
        tokens = list(filter(lambda t: t not in EMOJI_DICT, tokens))
        return tokens
    
    def _replace_emojis(self,
                        tokens):
        """
        Replace emojis with a generic <EMOJI> token
        
        Args:
            tokens (list): Input list of strings
        
        Returns:
            tokens (list): Tokens with emojis replace with generic string
        """
        tokens = list(map(lambda t: "<EMOJI>" if t in EMOJI_DICT else t, tokens))
        return tokens

    def tokenize(self,
                 text):
        """
        Tokenize a string of text. Can possibly be HTML formatted, as we
        perform normalization at the beginning. Tokenization will adhere
        to attributes chosen at the initialization of the class.

        Args:
            text (str): Input line of text.
        
        Returns:
            tokens (list of str): Tokenized version of the text.
        """
        ## Base case
        if text is None or len(text) == 0:
            return []
        ## Naively Filter Really long tokens (not particularly useful for modeling)
        ## These tokens also tend to cause the Twokenizer code to hang
        if any(l > 25 for l in list(map(len, text.split()))):
            text = " ".join([t for t in text.split() if len(t) < 25 or re.match(url, t)])
            if len(text) == 0:
                return []
        ## HTML Parsing
        text = normalize_text(text)    
        ## Get Tokens
        tokens = simpleTokenize(squeezeWhitespace(text))
        ## Upper Flag
        if hasattr(self, "upper_flag") and self.upper_flag:
            tokens = self._upper_flag(tokens)
        ## Numeric Values
        if hasattr(self, "keep_numbers") and not self.keep_numbers:
            tokens = self._strip_numbers(tokens)
        elif hasattr(self, "keep_numbers") and self.keep_numbers:
            tokens = self._replace_numbers(tokens)
        ## Hashtags
        if hasattr(self, "keep_hashtags"):
            if not self.keep_hashtags:
                tokens = self._remove_hashtag(tokens)
            else:
                tokens = self._clean_hashtag(tokens)
        ## Case Normalization
        if not hasattr(self, "keep_case") or not self.keep_case:
            tokens = list(map(lambda i: "<HASHTAG={}".format(i.replace("<HASHTAG=","").lower()) if i.startswith("<HASHTAG=") else i, tokens))
            tokens = list(map(lambda i: i.lower() if not i.startswith("<") and not i.endswith(">") else i, tokens))
        ## Retweet Tokens
        if hasattr(self, "keep_retweets"):
            if not self.keep_retweets:
                tokens = self._strip_retweets(tokens)
            else:
                tokens = self._replace_retweets(tokens)
        ## Strip User Mentions
        if hasattr(self, "keep_user_mentions"):
            if not self.keep_user_mentions:
                tokens = self._strip_user_mentions(tokens)
            else:
                tokens = self._replace_user_mentions(tokens)
        ## Contraction Expansion
        if not hasattr(self, "expand_contractions") or self.expand_contractions:
            tokens = self._expand_contractions(tokens)
        ## Negation handling
        if hasattr(self, "negate_handling") and self.negate_handling:
            tokens = self._handle_negations(tokens)
        ## Stopword Removal
        if hasattr(self, "stopwords"):
            tokens = list(filter(lambda x: x.lower().replace("not_","") not in self.stopwords, tokens))
        ## URL Handling
        if hasattr(self, "keep_url"):
            if not self.keep_url:
                tokens = self._strip_url(tokens)
            else:
                tokens = self._replace_url(tokens)
        ## Punctuation
        if not hasattr(self, "keep_punctuation") or not self.keep_punctuation:
            tokens = self._strip_punctuation(tokens)
        ## Emojis
        tokens = self._expand_emoji_groups(tokens)
        if hasattr(self, "emoji_handling") and self.emoji_handling is not None:
            if self.emoji_handling == "replace":
                tokens = self._replace_emojis(tokens)
            elif self.emoji_handling == "strip":
                tokens = self._strip_emojis(tokens)
            else:
                raise ValueError("emoji_handling should be 'replace', 'strip', or None.")
        return tokens
