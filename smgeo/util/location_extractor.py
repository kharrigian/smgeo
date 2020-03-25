
##########################
### Imports
##########################

## Standard Library
import os
import string
import re
import json

## External Libraries
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from unidecode import unidecode

## Local
from smgeo.util.tokenizer import (Tokenizer,
                                  get_ngrams,
                                  flatten)

##########################
### Helpers
##########################

def _safe_decode(text):
    """

    """
    if pd.isnull(text):
        return text
    return unidecode(text)

## Data Directory
DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../../data/"

##########################
### Geographic Data Resources
##########################

## Geonames Column Names
geoname_columns = [
"geonameid",
"name",
"asciiname",
"alternatenames",
"latitude",
"longitude",
"feature_class",
"feature_code",
"country_code",
"cc2",
"admin1_code",
"admin2_code",
"admin3_code",
"admin4_code",
"population",
"elevation",
"dem",
"timezone",
"modification_date",
]

## Cities with Population > 15,000 (Geonames)
city_file = f"{DATA_DIR}resources/cities15000.txt"
cities = pd.read_table(city_file,
                       header=None,
                       names=geoname_columns)

## Country Info
country_file = f"{DATA_DIR}resources/countryinfo.txt"
countries = pd.read_table(country_file)

## Admin Codes (1)
admin1_file = f"{DATA_DIR}resources/admin1CodesASCII.txt"
admin1 = pd.read_table(admin1_file,
                       header=None,
                       names=["code","name","name_ascii", "geonameid"])
admin1["ISO"] = admin1["code"].map(lambda i: i.split(".")[0])
admin1["admin1_code"]  = admin1["code"].map(lambda i: i.split(".")[1])

## Admin Codes (2)
admin2_file = f"{DATA_DIR}resources/admin2Codes.txt"
admin2 = pd.read_table(admin2_file,
                       header=None,
                       names=["code","name","name_ascii","geonameid"])
admin2["ISO"] = admin2["code"].map(lambda i: i.split(".")[0])
admin2["admin1_code"]  = admin2["code"].map(lambda i: i.split(".")[1])
admin2["admin2_code"]  = admin2["code"].map(lambda i: i.split(".")[2])

## Merge Resources
geo_resources = pd.merge(cities,
                         countries[["Country","Continent","ISO"]],
                         left_on=["country_code"],
                         right_on="ISO",
                         how="left").rename(columns={"Country":"country",
                                                     "Continent":"continent",
                                                     "name":"city",
                                                     "asciiname":"city_ascii"})
geo_resources = pd.merge(geo_resources,
                         admin1.drop(["code","geonameid"],axis=1),
                         left_on=["ISO","admin1_code"],
                         right_on=["ISO","admin1_code"],
                         how="left").rename(columns={"name":"state",
                                                     "name_ascii":"state_ascii"})
geo_resources = pd.merge(geo_resources,
                         admin2.drop(["code","geonameid"],axis=1),
                         left_on=["ISO","admin1_code","admin2_code"],
                         right_on=["ISO","admin1_code","admin2_code"],
                         how="left").rename(columns={"name":"county",
                                                     "name_ascii":"county_ascii"})

## Continent Formatting
geo_resources["continent"] = geo_resources["continent"].fillna("NA")
_continent_map = {'EU':"Europe",
                  'AS':"Asia",
                  'NA':"North America",
                  'AF':"Africa",
                  'SA':"South America",
                  'OC':"Oceania",
                  'AN':"Antartica"}
geo_resources["continent"] = geo_resources["continent"].map(lambda i: _continent_map[i])

## Missing Gazeteer Items
extra_cities_file = f"{DATA_DIR}resources/extra_cities.csv"
extra_cities = pd.read_csv(extra_cities_file)
geo_resources = geo_resources.append(extra_cities).reset_index(drop=True)

## Abbreviations
geo_abbr_file = f"{DATA_DIR}resources/abbreviations.csv"
geo_abbreviations = pd.read_csv(geo_abbr_file)

## Location Affixes
geo_affix_file = f"{DATA_DIR}resources/location_affixes.json"
geo_affixes = json.load(open(geo_affix_file, "r"))

## Country to Continent Map (Filling in Missing Countries)
country_continent_map = geo_resources.set_index("country")["continent"].to_dict()
country_continent_map["Antarctica"] = "Antarctica"
country_continent_map["South Georgia and the South Sandwich Islands"] = "Antarctica"
country_continent_map["French Southern Territories"] = "Antarctica"
country_continent_map['The Bahamas'] = "North America"
country_continent_map['Curaçao'] = "South America"
country_continent_map["Côte d'Ivoire"] = "Africa"
country_continent_map['Myanmar (Burma)'] = "Asia"
country_continent_map['Åland Islands'] = "Europe"
country_continent_map['Federated States of Micronesia'] = "Oceania"
country_continent_map['Cape Verde'] = "Africa"
country_continent_map['Palestine'] = "Asia"
country_continent_map['Vatican City'] = "Europe"

##########################
### Language Data Resources
##########################

## Frequently Ocurring Words (Note there are multiple word types per word token)
coca_file = f"{DATA_DIR}resources/coca5000.csv"
coca_words = pd.read_csv(coca_file, usecols=["word"])["word"].tolist()

## NLTK Stopwords
nltk_stopwords = stopwords.words("english")
nltk_stopwords = list(map(unidecode, nltk_stopwords))

## Additional Words to Ignore (Based on Trial + Error)
ignore_words_file = f"{DATA_DIR}resources/location_ignore_words.txt"
ignore_words = set(list(map(lambda i: i.strip(), open(ignore_words_file, "r").readlines())))

##########################
### Location Extraction
##########################

class LocationExtractor(object):

    """

    """

    def __init__(self):
        """

        """
        ## Class Initialization
        self._initialize_class_resources()
        ## Compile Dictionaries
        self._compile_gazeteer()
        self._compile_abbreviations()
        self._compile_geolocation_hierarchy()
    
    def __repr__(self):
        """

        """
        return "LocationExtractor()"
    
    def _initialize_class_resources(self):
        """

        """
        ## Geo Resources
        self._geo_resources = geo_resources.copy()
        ## Abbreviations
        self._geo_abbr = geo_abbreviations.copy()
        ## Affixes
        self._geo_affixes = geo_affixes.copy()
        ## Common Words
        self._common_words = set(coca_words) | \
                             set(nltk_stopwords) | \
                             set(ignore_words)
        ## Tokenizer
        self.tokenizer = Tokenizer(stopwords=None,
                                   keep_case=True,
                                   negate_handling=False,
                                   negate_token=False,
                                   upper_flag=False,
                                   keep_punctuation=True,
                                   keep_numbers=False,
                                   expand_contractions=False,
                                   keep_user_mentions=False,
                                   keep_pronouns=True,
                                   keep_url=False,
                                   keep_hashtags=False,
                                   keep_retweets=False,
                                   emoji_handling="strip")

    
    def _compile_gazeteer(self):
        """

        """
        ## Initialize Gazeteer
        self.gazeteer = set()
        ## Add City Names
        self.gazeteer.update(self._geo_resources["city_ascii"].map(_safe_decode).str.lower())
        ## Add County Names
        self.gazeteer.update(self._geo_resources["county_ascii"].map(_safe_decode).dropna().str.lower())
        ## Add State Names
        self.gazeteer.update(self._geo_resources["state_ascii"].map(_safe_decode).dropna().str.lower())
        ## Add Country Names
        self.gazeteer.update(self._geo_resources["country"].map(_safe_decode).dropna().str.lower())
        ## Add Continent Names
        self.gazeteer.update(self._geo_resources["continent"].map(_safe_decode).dropna().str.lower())
        ## Filter Out Common Words
        for cw in self._common_words:
            if cw in self.gazeteer:
                self.gazeteer.remove(cw)
        ## Filter Out Small Words
        self.gazeteer = set([i for i in self.gazeteer if len(i) > 3])

    def _compile_abbreviations(self):
        """

        """
        self.abbr_map = dict((y, x) for _, (x,y) in self._geo_abbr[["name","abbreviation"]].iterrows())

    def _compile_geolocation_hierarchy(self):
        """

        """
        self.geo_hierarchy = {"city":{},"county":{},"state":{},"country":{}}
        ## Country -> Continent
        for _, (country, continent) in self._geo_resources[["country","continent"]].iterrows():
            self.geo_hierarchy["country"][country.lower()] = set([continent.lower()])
        ## State -> Country, Continent
        for _, (state, country, continent) in self._geo_resources[["state_ascii","country","continent"]].iterrows():
            if pd.isnull(state):
                continue
            if state.lower() not in self.geo_hierarchy["state"]:
                self.geo_hierarchy["state"][state.lower()] = set()
            self.geo_hierarchy["state"][state.lower()].add(country.lower())
            self.geo_hierarchy["state"][state.lower()].add(continent.lower())
        ## County -> State, Country, Continent
        for _, (county, state, country, continent) in self._geo_resources[["county_ascii","state_ascii","country","continent"]].iterrows():
            if pd.isnull(county):
                continue
            if county.lower() not in self.geo_hierarchy["county"]:
                self.geo_hierarchy["county"][county.lower()] = set()
            if not pd.isnull(state):
                self.geo_hierarchy["county"][county.lower()].add(state.lower())
            self.geo_hierarchy["county"][county.lower()].add(country.lower())
            self.geo_hierarchy["county"][county.lower()].add(continent.lower())
        ## City -> County, State, Country, Continent
        for _, (city, county, state, country, continent) in self._geo_resources[["city_ascii","county_ascii","state_ascii","country","continent"]].iterrows():
            ca = city.lower()
            if ca not in self.geo_hierarchy["city"]:
                self.geo_hierarchy["city"][ca] = set()
            if not pd.isnull(county):
                self.geo_hierarchy["city"][ca].add(county.lower())
            if not pd.isnull(state):
                self.geo_hierarchy["city"][ca].add(state.lower())
            self.geo_hierarchy["city"][ca].add(country.lower())
            self.geo_hierarchy["city"][ca].add(continent.lower())

    def _filter_out_substrings(self,
                               strings):
        """

        """
        strings = sorted(set(strings), key=lambda x: len(x))
        filtered_strings = []
        n = len(strings)
        for i in range(n):
            matches_ahead = False
            str_i = strings[i]
            for j in range(i+1, n):
                if str_i in strings[j]:
                    matches_ahead = True
                    break
            if not matches_ahead:
                filtered_strings.append(str_i)
        return filtered_strings

    def _look_for_exact_match(self,
                              tokens):
        """

        """
        ## Get Lowercase N-Grams
        ngrams = get_ngrams([t.lower() for t in tokens], 1, 4)
        ngrams = list(map(lambda n: " ".join(list(n)), ngrams))
        ## Identify Exact Matches
        matches = []
        for n in ngrams:
            if n in self.gazeteer:
                matches.append(n)
        ## Remove Substrings
        matches_filtered = self._filter_out_substrings(matches)
        return matches_filtered

    def _combine_syntax_matches(self,
                                matches):
        """

        """
        n_matches = len(matches)
        if n_matches == 1:
            return matches
        combined_syntax_matches = []
        j = 0
        while j < n_matches - 1:
            match_j = matches[j].split(", ")
            k = j + 1
            while k < n_matches and matches[k].split(", ")[0] == match_j[-1]:
                match_j.append(matches[k].split(", ")[1])
                k += 1
            j = k
            combined_syntax_matches.append(", ".join(match_j))
        return combined_syntax_matches
        
    def _look_for_syntax_match(self,
                               tokens,
                               window = 4):
        """

        """
        n = len(tokens)
        j = 1
        syntax_matches = []
        while j < n - 1:
            if tokens[j] != ",":
                j += 1
            else:
                before_window = []
                after_window = []
                of_cache = ""
                for b in tokens[max(0, j-window):j][::-1]:
                    if b.istitle() and not b.startswith(":") or b.lower() == "of":
                        if b.lower() not in self._common_words:
                            before_window.append(b + of_cache)
                            of_cache = ""
                        elif b.lower() in self._geo_affixes["suffix"] and len(before_window) == 0:
                            before_window.append(b + of_cache)
                            of_cache = ""
                        elif b.lower() in self._geo_affixes["prefix"] and len(before_window) > 0:
                            before_window.append(b + of_cache)
                            of_cache = ""
                        elif b.lower() == "of":
                            of_cache += " of"
                        else:
                            of_cache = ""
                            break
                    else:
                        of_cache = ""
                        break
                before_window = before_window[::-1]
                if all(b.lower() in self._common_words or
                       b.lower() in self._geo_affixes["suffix"] or 
                       b.lower() in self._geo_affixes["prefix"] for b in before_window):
                    before_window = []
                of_cache = ""
                for a in tokens[j+1:min(j+1+window, n)]:
                    if (a.istitle() and not a.startswith(":")) or (a.lower() == "of"):
                        if a.lower() not in self._common_words:
                            after_window.append(of_cache + a)
                            of_cache = ""
                        elif a.lower() in self._geo_affixes["prefix"] and len(after_window) == 0:
                            after_window.append(of_cache + a)
                            of_cache = ""
                        elif a.lower() in self._geo_affixes["suffix"] and len(after_window) > 0:
                            after_window.append(of_cache + a)
                            of_cache = ""
                        elif a.lower() == "of":
                            of_cache += "of "
                        else:
                            of_cache = ""
                            break
                    else:
                        of_cache = ""
                        break
                if all(a.lower() in self._common_words or
                       a.lower() in self._geo_affixes["suffix"] or 
                       a.lower() in self._geo_affixes["prefix"] for a in after_window):
                    after_window = []
                if len(before_window) == 0 or len(after_window) == 0:
                    j += 1
                else:
                    same_level_match = False
                    for level in ["state","country"]:
                        if any(b.lower() in self.geo_hierarchy[level] for b in before_window) and \
                           any(a.lower() in self.geo_hierarchy[level] for a in after_window):
                           same_level_match = True
                    if not same_level_match:
                        before_window_comb = " ".join(before_window)
                        after_window_comb = " ".join(after_window)
                        if before_window_comb.lower() in self.gazeteer or \
                        after_window_comb.lower() in self.gazeteer:
                            syntax_matches.append(before_window_comb + ", " + after_window_comb)
                    j += len(after_window)
        ## Normalize Case
        syntax_matches = [s.lower() for s in syntax_matches]
        ## Combine Syntax Matches
        syntax_matches = self._combine_syntax_matches(syntax_matches)
        return syntax_matches

    def _expand_abbreviations(self,
                              tokens):
        """

        """
        expanded_tokens = [tokens[0]]
        for i in range(1, len(tokens)):
            if tokens[i-1] == "," and tokens[i].replace(".","") in self.abbr_map:
                expanded_tokens.append(self.abbr_map[tokens[i].replace(".","")])
            else:
                expanded_tokens.append(tokens[i])
        expanded_tokens = flatten(i.split(" ") for i in expanded_tokens)
        return expanded_tokens
    

    def _find_sub_list(self,
                       sl,
                       l):
        results = []
        sll = len(sl)
        for ind in (i for i,e in enumerate(l) if e==sl[0]):
            if l[ind:ind+sll]==sl:
                results.append((ind,ind+sll-1))
        return results

    def _append_affixes(self,
                        tokens,
                        matches):
        """

        """
        n = len(tokens)
        tokens_lower = list(map(lambda i: i.lower(), tokens))
        tokens_lower = flatten([i.split(" ") for i in tokens_lower])
        affixed_matches = []
        for m in matches:
            ## Re-Tokenize Match
            m_toks = self.tokenizer.tokenize(m)
            ## Look For Span Matches
            m_spans = self._find_sub_list(m_toks, tokens_lower)
            ## Check Spans for Affixes
            for span_start, span_end in m_spans:
                affixed_span = m_toks.copy()
                if span_start > 0:
                    if tokens[span_start-1].istitle() and tokens[span_start-1].lower() in self._geo_affixes["prefix"]:
                        affixed_span = tokens_lower[span_start-1:span_start] + affixed_span
                if span_end < n - 2:
                    if tokens[span_end+1].istitle() and tokens[span_end+1].lower() in self._geo_affixes["suffix"]:
                        affixed_span = affixed_span + tokens_lower[span_end+1:span_end+2]
                affixed_matches.append(" ".join(affixed_span).replace(" ,",","))
        return affixed_matches
    
    def _find_locations(self,
                        sent):
        """

        """
        ## Tokenize Sentence
        tokens = self.tokenizer.tokenize(sent)
        if len(tokens) == 0:
            return []
        ## Expand Abbreviations
        tokens = self._expand_abbreviations(tokens)
        ## Look for Exact Matches
        exact_matches = self._look_for_exact_match(tokens)
        ## Look for Syntax Matches
        syntax_matches = self._look_for_syntax_match(tokens)
        ## Consolidate Matches
        combined_matches = self._filter_out_substrings(syntax_matches + exact_matches)
        ## Append Affixes
        affixed_matches = self._append_affixes(tokens, combined_matches)
        return affixed_matches

    def _merge_overlap(self,
                       matches):
        """

        """
        merged_matches = []
        seen = set()
        for m, match in enumerate(matches):
            match_split = match.split(", ")
            match_set = [match]
            seen.add(m)
            for m2, match2 in enumerate(matches):
                if m == m2 or m2 in seen:
                    continue
                match2_split = match2.split(", ")
                if set(match_split) & set(match2_split) != set():
                    match_set.append(match2)
                    seen.add(m2)
                    seen_added = True
            match_set = self._combine_syntax_matches(match_set)
            merged_matches.extend(match_set)
        return self._filter_out_substrings(merged_matches)

    def _combine_using_hierarchy(self,
                                 matches):
        """

        """
        combined_matches = []
        reg_matches = []
        for m, match in enumerate(matches):
            match_split = match.split(", ")
            match_level = None
            proposed_match = None
            for m2, match2 in enumerate(matches):
                if m == m2:
                    continue
                match2_split = match2.split(", ")
                levels = ["city","county","state","country"]
                for tl, topy_level in enumerate(levels):
                    if match_split[-1] in self.geo_hierarchy[topy_level]:
                        if match2_split[0] in self.geo_hierarchy[topy_level][match_split[-1]]:
                            if match_level is None or topy_level in levels[:match_level]:
                                proposed_match = (m, m2)
                                match_level = tl
            if proposed_match is None:
                reg_matches.append(match)
            else:
                combined_matches.append((proposed_match, match_level))
        combined_matches = sorted(combined_matches, key=lambda x: x[1])
        for (ml, mr), _ in combined_matches:
            reg_matches.append(matches[ml] + ", " + matches[mr])
        ## Filter Duplicates
        reg_matches = self._filter_out_substrings(reg_matches)
        ## Merge Overlap
        reg_matches = self._merge_overlap(reg_matches)
        return reg_matches

    def _is_all_commons(self,
                        match):
        """

        """
        if all(m in self._common_words for m in match.replace(", ","").split()):
            return True
        return False

    def find_locations(self,
                       text):
        """

        """
        ## Translate to Ascii
        text = unidecode(text)
        ## Split Up Sentences
        sentences = sent_tokenize(text)
        ## Look for Locations
        locations = []
        for sent in sentences:
            sent_locs = self._find_locations(sent)
            locations.extend(sent_locs)
        ## Combine Using Hierarchy
        locations = self._combine_using_hierarchy(locations)
        ## Filter Commons
        locations = [l for l in locations if not self._is_all_commons(l)]
        return locations

# #######################
# ### Test Cases
# #######################

# ## Write out Test Cases
# test_cases = [
#     ("I grew up in Los Angeles, CA and then moved to Boston last september. Big fan of massachusetts", ["los angeles, california", "boston, massachusetts"]),
#     ("Currently living in the northeastern US. New England as some would call it.",["new england"]),
#     ("U.S. of A baby!", []),
#     ("Stockholm, Sweden, Europe.", ["stockholm, sweden, europe"]),
#     ("Born in the Fake City, Scotland, Everdeen",["fake city, scotland, everdeen"]),
#     ("Not a big fan of Synthetic City, ID", ["synthetic city, idaho"])
# ]

# ## Initialize Extractor
# le = LocationExtractor()

# ## Run Test Cases
# for t in test_cases:
#     print(t[0])
#     print("\t",le.find_locations(t[0]))

