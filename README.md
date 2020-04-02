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

## Installation

All code was developed using Python 3.7+. Core functionality is contained within the `smgeo` package, processes such as data acquisition, annotation, and model training are related to the `scripts/` directory. We expect all code to be run from the root directory of this repository. 

To run any of the scripts, you will need to install the `smgeo` package first. We recommend installing the pacakge using developer mode:

```
pip install -e .
```

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

Reference directories are specified in `configurations/settings.json`. All of the directories included in the default file are created by default when you clone this directory. If you instead want to store data in different directories, you can specify so within this file.

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
pytest tests/ --cov=mhlib/ --cov-report=html -v -Wignore
```
