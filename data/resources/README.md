# Resources

Many of the files in this directory come from external sources, which we shall credit below.

## Geonames Database

Files used to construct the gazeteer used in location extraction are source from the Geonames database (http://download.geonames.org/export/dump/). They were acquired on 3/10/2020, so a current pull might have slight differences.

* `admin1CodesASCII.txt`: State-level codes
* `admin2Codes.txt`: Count-level codes
* `cities15000.txt`: List of cities with a population greater than 15,000
* `countryInfo.txt`: Country-level codes

## Corpus of Contemporary English

Stopwords used to filter out potential city candidates during location extraction come from the Corpus of Contemporary English. In particular, we use the free list of the top 5000 most frequent tokens in the dictionary. The source is here https://www.wordfrequency.info/free.asp. Please consider downloading directly from the website so that they have a record of the acquisition.

* `coca5000.csv`: Top 5000 most frequent tokens in the Corpus of Contemporary English.

## Google Region Codes

Mapping between 2-letter region codes and countries used by the Google Geocoding API. Comes from https://en.wikipedia.org/wiki/ISO_3166-1.

* `google_region_codes.csv`: Region code to country mapping.

## Additional Location Extraction Resources (Manually curated)

Various files were manually curated to aid in the location identification process.

* `location_affixes.json`: Common location name prefixes and suffixes.
* `location_ignore_words.txt`: Additional words to add to the stopword list for location filtering.
* `extra_cities.csv`: Cities not included in the Geonames data set that were commonly seen in a qualitative review of the seed data.
* `abbreviations.csv`: Common location abbreviations (US States, Canadian Provinces, and the UK)