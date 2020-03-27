

## Parameters
MIN_RESOLUTION = "locality"
N_BINS = 20

## Directories
LABELS_DIR = "./data/raw/reddit/labels/"
AUTHORS_DIR = "./data/raw/reddit/authors/"
PLOT_DIR = "./plots/"

####################
### Imports
####################

## Standard Library
import os
import json
import gzip
from datetime import datetime, timezone
from collections import Counter
from multiprocessing import Pool

## External Libraries
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Local Modules

####################
### Helpers
####################

def filter_labels_by_resolution(labels,
                                min_resolution):
    """

    """
    resolutions = ["locality",
                   'administrative_area_level_3',
                   'administrative_area_level_2',
                   'administrative_area_level_1',
                   'country',
                   'continent']
    labels_res = labels[resolutions].isnull().idxmin(axis=1)
    acceptable_res = []
    for r in resolutions:
        acceptable_res.append(r)
        if r == min_resolution:
            break
    labels_filtered = labels.loc[labels_res.isin(set(acceptable_res))]
    labels_filtered = labels_filtered.reset_index(drop=True).copy()
    return labels_filtered

def convert_epoch_to_local_time(epoch):
    """

    """
    ## Make Epoch to Datetime Conversion
    conversion = datetime.utcfromtimestamp(epoch)
    ## Make UTC to Local Time Timezone Conversion
    conversion = conversion.replace(tzinfo=timezone.utc).astimezone(tz=None)
    return conversion

def get_author_timestamp_distribution(author):
    """

    """
    ## Standard Path
    author_data_file = f"{AUTHORS_DIR}{author}.json.gz"
    ## No Data Exists
    if not os.path.exists(author_data_file):
        return (author, None)
    ## Load File
    with gzip.open(author_data_file, "r") as the_file:
        author_data = json.load(the_file)
    ## Parse Timestamps
    timestamps = list(map(convert_epoch_to_local_time,
                          list(map(lambda i: i["created_utc"], author_data))))
    ## Get Hour of Day Distribution
    hours = Counter(list(map(lambda x: x.hour, timestamps)))
    hours = list(map(lambda h: hours[h], range(24)))
    return (author, hours)

def assign_value_to_bin(value, bins):
    """

    """
    b = 0
    for bstart, bend in zip(bins[:-1], bins[1:]):
        if value >= bstart and value < bend:
            return b
        b += 1
    return b

####################
### Main Program
####################

def main():
    """

    """
    ## Load Labels
    label_file = f"{LABELS_DIR}author_labels.json.gz"
    labels = pd.read_json(label_file)
    ## Filter Down to Authors Meeting Resolution Criteria
    labels = filter_labels_by_resolution(labels, MIN_RESOLUTION)
    ## Load Hour Distributions
    mp = Pool(8)
    hours = list(tqdm(mp.imap_unordered(get_author_timestamp_distribution, labels["author"].tolist()), total=len(labels)))
    mp.close()
    ## Format Hour Distributions
    hours = list(filter(lambda r: r[1] is not None, hours))
    hours = pd.DataFrame(index =list(map(lambda i: i[0], hours)),
                       data = list(map(lambda i: i[1], hours)))
    ## Append Author Location
    hours = pd.merge(hours,
                     labels.set_index("author")[["longitude","latitude"]],
                     left_index=True,
                     right_index=True,
                     how = "left")
    ## Establish Longitude Bins
    lon_bounds = int(hours["longitude"].min() - 1), int(hours["longitude"].max() + 1)
    lon_bins = np.linspace(lon_bounds[0], lon_bounds[1], N_BINS + 1)
    ## Assign Longitudes to Bins
    hours["longitude_bin"] = hours["longitude"].map(lambda i: assign_value_to_bin(i, lon_bins))
    ## Create Comment Array
    time_array = np.zeros((N_BINS, 24))
    for b in range(N_BINS):
        time_array[b] += hours.loc[hours["longitude_bin"] == b][list(range(24))].sum(axis=0).values
    ## Normalize
    comment_percent_array = time_array / time_array.sum(axis=1, keepdims=True)
    ## Plot
    fig, ax = plt.subplots(figsize=(10,5.8))
    mat = ax.imshow(comment_percent_array,
                    aspect = "auto",
                    cmap=plt.cm.Purples,
                    interpolation = "bilinear")
    ax.set_yticks(np.arange(0, N_BINS + 1) - 0.5)
    ax.set_yticklabels(["{:.0f}".format(l) for l in lon_bins])
    ax.set_ylim(N_BINS-.5, -.5)
    ax.set_xlabel("Hour of Day (US/Eastern)")
    ax.set_ylabel("Longitude Bin")
    cbar = fig.colorbar(mat)
    cbar.set_label("Proportion of Bin Comments")
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}hours_vs_longitude_res-{MIN_RESOLUTION}_bins-{N_BINS}.png", dpi=300)
    plt.close()

####################
### Execute
####################

if __name__ == "__main__":
    main()