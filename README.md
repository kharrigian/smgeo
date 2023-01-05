# Geolocation Inference for Social Media

This repository provides the first (to my knowledge) geolocation inference approach for Reddit. Over time, it will completely reproduce and extend results from "Geocoding without Geotags: A Text-based Approach for Reddit" (Harrigian 2018). This README contains information for running pretrained inference models and training your own. Please note that commerical use of any code or data in this repository is strictly prohibited. If you have any questions or are interested in contributing, please feel free to reach out to Keith Harrigian at kharrigian@jhu.edu. If you use any code or data, please cite the original paper using the bibliographic information below.

```
@inproceedings{harrigian2018geocoding,
  title={Geocoding Without Geotags: A Text-based Approach for reddit},
  author={Harrigian, Keith},
  booktitle={Proceedings of the 2018 EMNLP Workshop W-NUT: The 4th Workshop on Noisy User-generated Text},
  pages={17--27},
  year={2018}
}
```

Please note that use of code or data in this repository is governed by the LICENSE file and Data Usage Agreement. If you plan to use this code, please fill out the Data Usage Agreement and send it to Keith Harrigian at kharrigian@jhu.edu.

## Disclosure

*Important* As of January 1st, 2023 it is apparent that the backend API I am using for acquiring Reddit data is no longer working. Although I hope to update the codebase to support the new API in the future, I unfortunately don't have the bandwidth to do so at the moment. I recommend using the public dumps of data from Pushshift.io as opposed to the ElasticSearch backend API moving forward, as the latter tends to deal with severe rate limiting and extensive downtime.

## Installation

All code was developed using Python 3.7+. Core functionality is contained within the `smgeo` package, processes such as data acquisition, annotation, and model training are related to the `scripts/` directory. We expect all code to be run from the root directory of this repository. 

To run any of the scripts, you will need to install the `smgeo` package first. We recommend installing the pacakge using developer mode:

```
pip install -e .
```

## Models and Data

To download pretrained models and a subset of relevant data (e.g. user labels, geocoded strings), please fill out the Data Usage Agreement located in the root directory of this repository and send it to kharrigian@jhu.edu. The request will be generally be reviewed within 24 hours. Upon approval, you will receive a dropbox link where you can download the models and data. Note that we do not distribute any raw Reddit comment/submission data as part of this dump; if you plan on training your own models from scratch, it is recommended that you follow the instructions listed [below](#training).

While we provide the code in an open source manner that would theoretically enable full reproduction of our models and data assets, we request that you still fill out a data usage agreement. We simply want to ensure that this code base is not used for any malicious reasons and that any user of code/data in this repository is aware of its distribution constraints.

#### Data

This repository houses the bare minimum data to recreate the training process from scratch (e.g. seed submissions, subreddit biases, location gazeteer). Included as well in the dropbox is the following:

* `data/raw/reddit/labels/author_labels.json.gz`: Annotated Reddit user locations. Includes the comment or submission title which was used as a self-identified location disclosure, along with the extracted locations and geocodings. Note that because of the distant-supervision process we use for labeling, it is likely that this data is noisy. 
* `data/raw/reddit/google_geocoder_results.json`: Mapping between (location string, region bias) to result from the Google Geocoding API. 
* `data/raw/reddit/labels/seed_submission_comments_2020-02-26.csv`: Comments from the set of seed submissions used to annotate user locations.
* `data/raw/reddit/labels/submission_titles_2020-02-26.csv`: Submissions from r/AmateurRoomPorn used to annotate user locations.

#### Models

We will provide two pretrained geolocation inference models upon request. In addition to the raw ".joblib" model file, we provide the configuration file used to train the model and the non-localness computations used to perform feature selection.

* `models/reddit/US_TextSubredditTime/`: Contains model and associated data for inferring location of users in the contiguous United States. This model is useful if you are confident your sample of Reddit users lives within the contiguous United States.
* `models/reddit/Global_TextSubredditTime/`: Contains model and associated data for inferring locations of users around the entire world. This model is useful if you are not confident your sample of Reddit users only lives within the contiguous United States.

Note that the model architecture used for geolocating Reddit users is outdated relative to methods used in state-of-the-art research. Adding modern (i.e. neural, network-based) approaches in on the to-do list for this project.

## Configuration

Users of this package need to configure two files before running any code in this repository. Templates have been provided to make this setup easy.

#### API Keys

If you want to query any data using the official Reddit API (PRAW) or geocode new strings using the Google Cloud Geocoding API, you will need to provide valid credentials. Credentials should be stored in a file called `config.json` housed in the root directory of this repository. A template file has been included and is replicated below. Note that Reddit API credentials are not necessary for all tasks (e.g. pulling comment data or searching for submissions). Instead, one can opt to query data from Pushshift.io using the PSAW wrapper. Google Cloud API credentials are only necessary if you plan to geocode location strings.

```
{
    "reddit": {
        "client_id": "CLIENT_ID",
        "client_secret": "CLIENT_SECRET",
        "username": "REDDIT_USERNAME",
        "password": "REDDIT_PASSWORD",
        "user_agent": "APP_USER_AGENT"
    },
    "google":{
        "api_key":"API_KEY"
    }
}
```

#### Settings (Directories)

Reference directories are specified in `configurations/settings.json`. All of the directories included in the default file are created by default when you clone this directory. If you instead want to store data in different directories, you can specify so within this file. In general, we recommend keeping this structure to make understanding references within scripts easy.

```
{   
    "reddit":{
            "LABELS_DIR":"./data/raw/reddit/labels/",
            "AUTHORS_PROCESSED_DIR":"./data/processed/reddit/authors/",
            "DATA_CACHE_DIR":"./data/processed/reddit/features/",
            "MODELS_DIR":"./models/reddit/",
            "AUTHORS_RAW_DIR":"./data/raw/reddit/authors/"
            }
}
```

## Testing

A basic test suite has been developed to ensure that code behaves consistently over time. The test suite uses `pytest` and `pytest-cov`. We recommend running the test suite after installation of the `smgeo` package to make sure everything was installed appropriately. You can run all tests using the following command:

```
pytest tests/ -Wignore -v
```

Alternatively, to run tests and check package coverage, you can do so using

```
pytest tests/ --cov=smgeo/ --cov-report=html -v -Wignore
```

## Inference

To infer the location of Reddit users using one of our pretrained models, you can use the following basic wrapper command:

```
python scripts/model/reddit/infer.py <model_path> <user_list> <output_csv>
```

* `<model_path>`: Path to a "model.joblib" file. See README in "./models/" for information about available models.
* `<user_list>`: Path to a ".txt" file with one Reddit username per line. The code will pull comment histories for these users if they haven't already been queried.
* `<output_csv>`: Name of a ".csv" file for storing the inferences.

There are several customizable parameters to include when running the inference script. You can see them by running the following command (or reading further).

```
python scripts/model/reddit/infer.py --help
```

#### Inference Arguments

* `--overwrite_existing_histories` - By default, post histories for a Reddit user are collected once and cached for future use. If this argument is included, a new query will be made to collect recent comment data.
* `--start_date` - ISO-format string representing the start date for collecting user comment data. Default is "2008-01-01". Recent data more be indicative of a person's language if they have moved.
* `--end_date` - ISO-format string representing the end data for collecting user comment data. Default is the current date. Filtering out new data could be useful if analyzing historical posts.
* `--comment_limit` - Integer representing the maximum number of comments collected for each user. Default is 250. Decreasing speeds up query time, but risks diminishing classification performance.
* `--min_comments` - Integer specifying the minimum number of comments found in a user's history to qualify for inference. In general, users with more comments (>50) have more accurate inferences.
* `--grid_cell_size` - Float specifying the size in degrees of each grid cell to make inferences over. Only relevant if not using the `--known_coordinates` flag.
* `--posterior` - If this flag is included, the posterior over all coordinates considered by the model will be output for each user.
* `--reverse_geocode` - If included, the argmax of predictions will be assigned nominal geographic information (e.g. city, state, country)
* `--known_coordinates` - If specified (and you have access to the training label set), this will restrict inference coordinates to those seen during training time instead of using a standard grid of coordinates. Useful for identifying known cities as opposed to general regions.

## Training

If you are just interested in applying pretraned inference moels, the information contained above should be enough. However, if you are interested in understanding the full process that enabled us to train these models, feel free to keep reading.

### Step 1: Identify Seed Data

The major contribution of the original W-NUT paper was showing that one can use distant supervision to train geolocation inference models for Reddit. This is required because, unlike Twitter, Reddit does not have geo-tracking features as part of the platform. Moreover, Reddit users are pseudoynmous and are not required to link any of their offline identify with their online identity.

Our solution to this problem was using self-identified disclosures of home location in a carefully curated set of Reddit submissions. In the original paper (and here as well), we identify roughly 1,400 submissions with titles similar to "Where are you from?" or "Where are you living?". In this updated version, we also use submissions in the r/AmateurRoomPorn subreddit, where many posts include the location of the room in the title.

We search for submissions using `scripts/acquire/reddit/search_for_submissions.py`. This script generates a CSV of submission titles relevant to the aforementioned questions. The output of our query is found in `data/raw/reddit/labels/submission_candidates.csv`. This file was manually reviewed, with submissions that were unlikely to contain self-identified locaton disclosures marked to be ignored, alongside submissions with less than 50 comments. The result of the manual curation is found in `data/raw/reddit/labels/submission_candidates_filtered.py`.

With the seed submissions identified, we then queried the comments from all relevant threads. Additionally, we queried all submissions from the r/AmateurRoomPorn subreddit. This was done using the `scripts/acquire/reddit/retrieve_self_identification_data.py` script. It shouldn't take long to requery this data. However, feel free to reachout to kharrigian@jhu.edu if you'd like the exact files used to train our pretrained models.

### Step 2: Annotate Users

With the seed data queried, we now want to associate Reddit users with their self-identified location disclosures. Our queries provide thousands of users and comments which would be infeasible to annotate by hand. Thus, we instead use a programmatic named-entity-recognition (NER) approach to identify disclosures in the comments and submission titles. This process is carried out in `scripts/annotate/reddit/extract_locations.py`. 

The NER approach we use is based on exact matches to a gazeteer and other syntax-based rules. Once the location strings are extracted, we use the Google Geocoding API to assign likely coordinates for the location string. To bias the queries, we manually curate a mapping between subreddits and regions. For example, r/CasualUK is associated with Great Britain, and thus a mention of Scarborough in this subreddit is more likely to reference the city in England as opposed to Ontario Canada. The mapping of biases used for this iteration of the model is found in `data/raw/reddit/labels/subreddit_biases.csv`.

### Step 3: Query User Comments and Preprocess

Perhaps the most straightforward part of the training process is querying comment histories and processing them to be in a machine-readable format.

Comment histories are queried from the Pushshift.io database using `scripts/acquire/reddit/retrieve_author_histories.py`. Raw histories are cached to disk for future use.

Author histories are tokenized and cached for future use using `scripts/preprocess/reddit/prepare_reddit.py`. Caching of these preprocessed comment histories make vectorization for our models more efficient moving forward.

### Step 4: Inference Model Training

Inference models can be trained using `scripts/model/reddit/train.py`. The input to this script includes paths to a data configuration and a model configuration file. The syntax is as follows:

```
python scripts/model/reddit/train.py configurations/data/<data_config>.json configurations/data/<model_config>.json
```

##### Data Configuration

Data set configurations are housed in `configurations/data/`. The base template included in this repository can be manipulated if desired (and you have access to preprocessed data). The cached data will be stored in directory specified by the `DATA_CACHE_DIR` parameter in `settings.json`.

```
{
    "NAME":"state_min5comments_250docs",
    "MIN_RESOLUTION":"administrative_area_level_1",
    "MIN_COMMENTS":5,
    "VOCAB_PARAMETERS":{
                        "text":true,
                        "subreddits":true,
                        "time":true,
                        "text_vocab_max":null,
                        "subreddit_vocab_max":null,
                        "min_text_freq":10,
                        "max_text_freq":null,
                        "min_subreddit_freq":10,
                        "max_subreddit_freq":null,
                        "max_toks":null,
                        "max_docs":250,
                        "binarize_counter":true
                        }
}
```

The parameters are as follows:
* `NAME`: Name of the cache. Arbitrarily assigned to make it easy to keep track of preprocessed data.
* `MIN_RESOLUTION`: Users have been labeled at various resoulutions (e.g. locality, country). This parameter specifies the minimum resolution to require for sampling users. For example, "administrative_area_level_1" would include users labeled at both a "locality" and "administrative_area_level_1" level.
* `MIN_COMMENTS`: Minimum number of comments in the user's comment history to be kept in the data set.
* `VOCAB_PARAMETERS`: Passed to the `Vocabulary` class as kwargs for learning the base vocabulary. It's recommended to lean on the side of having too large a vocabulary for this stage and then filtering down using the model parameters.
* `text`: If True, include bag-of-words text features.
* `subreddits`: If True, include bag-of-words subreddit features (e.g. what subreddits did the user post comments in).
* `time`: If True, include the distribution of comments by hour of day (UTC).
* `text_vocab_max`: Maximum number of text tokens to keep as features (most common)
* `subreddit_vocab_max`: Maximum number of subreddits to kee as features (most common)
* `min_text_freq`: Minimum occurrence of a text token in the data set.
* `max_text_freq`: Maximum occurrence of a text token in the data set.
* `min_subreddit_freq`: Minimum occurrence of a subreddit feature in the data set.
* `max_subreddit_frew`: Maximum occurrence of a subreddit feature in the data set.
* `max_toks`: Select the first *n* tokens from each comment.
* `max_docs`: Select the *n* most recent comments from the user history.
* `binarize_counter`: If True, the occurence of a token is only counted toward the total thresholds once per user. Otherwise, multiple uses by a single user count seprately in the total count.

##### Model Configuration

This file specified how you want the inference model to be trained. Model configurations are housed in `configurations/model/`. The base template included in this repository can be manipulated if desired. Configuratons for our pretrained models have been included as well for reference.

```
{
 "NAME":"Global_TextSubreddit",
 "FILTER_TO_US":false,
 "MIN_MODEL_RESOLUTION": "administrative_area_level_1",
 "TOP_K_TEXT":50000,
 "TOP_K_SUBREDDITS":1100,
 "MIN_SUPPORT":25,
 "USE_TEXT":true,
 "USE_SUBREDDIT":true,
 "USE_TIME":false,
 "RANDOM_STATE":42
}
```

The parameters are as follows:
* `NAME`: What the model file will be called.
* `FILTER_TO_US`: If True, model training is restricted to users from the contiguous United States.
* `MIN_MODEL_RESOLUTION`: Similar to `MIN_RESOLUTION` in the data configuration file. This is your chance to further restrict the user pool for training.
* `TOP_K_TEXT`: Number of text features to include in the model, selected using Non-localness.
* `TOP_K_SUBREDDITS`: Number of subreddit features to include in the model, selected using Non-localness.
* `MIN_SUPPORT`: Minimum frequency of a feature across users to include in the model.
* `USE_TEXT`: If True, use text features in the model (if available in the data cache).
* `USE_SUBREDDIT`: If True, use subreddit features in the model (if available in data cache).
* `USE_TIME`: If True, use temporal features in the model (if available in the data cache).
* `RANDOM_STATE`: Random seed for training the model.

## Dataset Noise

It is important to keep in mind that our "ground truth" dataset of user geolocation labels is based on an inherently noisy distant supervision process. User location labels used for training models may be wrong and inherently skew training/evaluation. Furthermore, models are trained on data sampled around the date of self-disclosure for each user, but may not perfectly capture the most relevant data in cases where users moved within our sample window. 

#### Annotation Evaluation

With respect to annotation precision, an analysis of a random sample suggests the dataset contains only limited amounts of noise (see below). Moreover, the errors that arise during the annotation process are largely a result of overly-conservative labeling rules. Those interested in improving the distant geocoding procedure should reach out using the contact information listed above.

```
Label Precision: 488/500 (97.600%)
Resolution Precision: 428/488 (87.705%)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Incorrect Error Analysis:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
disambiguation -> 7
parsing -> 5
movement -> 1
multiple_locations -> 1
irrelevant_mention -> 1
geocoder_error -> 1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Correct Error Analysis:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
multiple_locations -> 22
geocoding_specificity -> 18
parsing -> 9
stopwords -> 4
linking -> 3
geocoder_error -> 2
ambiguous -> 1
reverse_syntax -> 1
ner_abbreviation -> 1
gazeteer -> 1
unicode_handling -> 1
```

