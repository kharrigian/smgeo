
"""
Automatic Annotation Analysis (e.g. Precision)
"""

## Path to Labels Directory
LABEL_DIR = "data/raw/reddit/labels/"
ANNOTATION_FILE = "annotation_evaluation_sample_500_2020-07-25.csv"
SAMPLE_SIZE = 500

## Sample Seed
RANDOM_SEED = 42

######################
### Imports
######################

## Standard Library
import os

## External Libraries
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

## Local
from smgeo.util.helpers import flatten

######################
### Load/Prepare Labels
######################

## Label File
label_file = f"{LABEL_DIR}author_labels.json.gz"
labels = pd.read_json(label_file)

## Sort Data
labels = labels.sort_values("author", ascending=False).reset_index(drop=True)

## Format Locations List
loc_formatter = lambda x: "; ".join([f"<{i}>" for i in x])
labels["locations_formatted"] = labels["locations"].map(loc_formatter)

## Isolate Relevant Subset
labels_subset = labels[["author",
                        "id",
                        "link_id",
                        "source",
                        "subreddit",
                        "text",
                        "locations_formatted",
                        "locality",
                        "administrative_area_level_3",
                        "administrative_area_level_2",
                        "administrative_area_level_1",
                        "country",
                        "continent"]].rename(columns={
                            "locality":"geocoded_city",
                            "administrative_area_level_3":"geocoded_subcounty",
                            "administrative_area_level_2":"geocoded_county",
                            "administrative_area_level_1":"geocoded_state",
                            "country":"geocoded_country",
                            "continent":"geocoded_continent"
                        }).copy()

## Add Evaluation Columns
eval_cols = ["label_correct",
             "true_resolution",
             "geocoded_resolution",
             "true_city",
             "true_subcounty",
             "true_county",
             "true_state",
             "true_country",
             "true_continent",
             "error_reason"]
for col in eval_cols:
    labels_subset[col] = None

######################
### Dump/Load Sample
######################

## Select Sample
annot_sample_file = f"{LABEL_DIR}{ANNOTATION_FILE}"
if not os.path.exists(annot_sample_file):
    labels_subset_sample = labels_subset.sample(SAMPLE_SIZE,
                                                replace=False,
                                                random_state=RANDOM_SEED).reset_index(drop=True)
    labels_subset_sample.to_csv(annot_sample_file, index=False)
else:
    labels_subset_sample = pd.read_csv(annot_sample_file)

## Check Evaluation
if labels_subset_sample["label_correct"].isnull().all():
    print("Sample data has not yet been evaluated. Exiting.")
    exit()

## Parsing Error Set
labels_subset_sample["error_reason"] = labels_subset_sample["error_reason"].map(lambda i: sorted(set(i.split(", "))) if \
                                                                                 not pd.isnull(i) else [])

######################
### Annotation Evaluation
######################

## Resolution Match
labels_subset_sample["resolution_correct"] = (labels_subset_sample["true_resolution"]==labels_subset_sample["geocoded_resolution"]).astype(int)

## Isolate Correctness Subsets
incorrect_df = labels_subset_sample.loc[labels_subset_sample["label_correct"]==0].copy()
correct_df = labels_subset_sample.loc[labels_subset_sample["label_correct"]==1].copy()

## Compute Precision Values
print("Label Precision: {}/{} ({:.3f}%)".format(correct_df.shape[0],    
                                                labels_subset_sample.shape[0],
                                                labels_subset_sample["label_correct"].mean()*100))
print("Resolution Precision: {}/{} ({:.3f}%)".format(
                                                correct_df["resolution_correct"].sum(),
                                                correct_df.shape[0],
                                                correct_df["resolution_correct"].mean()*100))

## Confusion Matrix
resolutions = ["neighborhood","city","subcounty","county","state","country","continent","region"]
cm = metrics.confusion_matrix(correct_df["true_resolution"],
                              correct_df["geocoded_resolution"],
                              labels=resolutions)
cm = pd.DataFrame(cm, index=resolutions, columns=resolutions)

## Visualize Confusion Matrix
fig, ax = plt.subplots(figsize=(8,5))
m = ax.imshow(cm.apply(lambda row: row / sum(row), axis=1),
              interpolation="nearest",
              cmap=plt.cm.Purples,
              aspect="auto",
              alpha=0.7)
for i, row in enumerate(cm.values):
    for j, val in enumerate(row):
        ax.text(j, i, val if val > 0 else "", color="black", va="center", ha="center", fontweight="bold")
cbar = fig.colorbar(m)
cbar.set_label("Resolution Precision", fontweight="bold", fontsize=14, labelpad=-54)
ax.set_xticks(list(range(cm.shape[1])))
ax.set_yticks(list(range(cm.shape[0])))
ax.set_xticklabels([r.title() for r in resolutions], fontsize=10, rotation=45, ha="right")
ax.set_yticklabels([r.title() for r in resolutions], fontsize=10)
for i in range(cm.shape[0]-1):
    ax.axvline(i+.5, color="black", alpha=.8)
    ax.axhline(i+.5, color="black", alpha=.8)
ax.set_ylabel("True Resolution", fontweight="bold", fontsize=14)
ax.set_xlabel("Geocoded Resolution", fontweight="bold", fontsize=14)
fig.tight_layout()
fig.savefig("./plots/annotation_resolution_confusion_matrix.png", dpi=300)
plt.close(fig)

## Error Analysis
for df, df_name in zip([incorrect_df, correct_df],["Incorrect","Correct"]):
    print("~"*35 + f"\n{df_name} Error Analysis:\n" + "~"*35)
    for reason, count in pd.Series(flatten(df["error_reason"])).value_counts().items():
        print(reason, "->", count)
