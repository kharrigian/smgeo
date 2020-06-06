
"""
Post-hoc analysis of cross-validation results
"""

#########################
### Configuration
#########################

## Name of Analysis
ANALYSIS_NAME = "Global_Feature_Modality"

## Directories
RESULTS_DIR = "./data/results/reddit/cross_validation/"
PLOTS_DIR = "./plots/"

## Cross-Validation Runs (Folder name, Label)
CV_DIRECTORIES = [
                ("2020_04_01_21_01_Global_Text", "Text"),
                ("2020_04_01_21_02_Global_TextSubreddit", "Text +\nSubreddits"),
                ("2020_04_01_21_03_Global_TextSubredditTime", "Text +\nSubreddits +\nTime")
]

## Analysis Parameters
ANNOTATE_PLOTS = False
CI_SAMPLE_PERCENT = 30
CI_N_SAMPLES = 100
CI_ALPHA = 0.05

#########################
### Imports
#########################

## Standard Library
import os
import sys
import json
from glob import glob

## External Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm import tqdm
import reverse_geocoder as rg # pip install reverse_geocoder

## Local Modules
from smgeo.util.logging import initialize_logger

## Initialize Logger
LOGGER = initialize_logger()

#########################
### Functions
#########################

def reverse_search(coordinates):
    """
    Use the Geonames Database to Reverse Search Locations based on Coordinates

    Args:
        coordinates (2d-array): [Lon, Lat] values
    
    Returns:
        result (list of dict): Reverse search results
    """
    result = rg.search(list(map(tuple,coordinates[:,::-1])))
    return result

def bootstrap_confidence_interval(x1,
                                  x2=None,
                                  func=np.mean,
                                  samples=100,
                                  sample_percent=30,
                                  alpha = 0.05):
    """
    Compute a statistic using a bootstrapped confidence interval

    Args:
        x1 (array): Array of data
        x2 (array or None): Optional second array of data (e.g. for classification scores)
        func (callable): Score function. Default is mean
        samples (int): Number of random samples to use
        sample_percent (float [0,100]): Percent of samples to use for each bootsrap calculation
        alpha (float [0,0.5]): Type 1 error rate. E.g. 0.05 returns 95% confidence interval around stat
    
    Returns:
        ci (array): [Lower, Median, Upper] statistics
    """
    ## Check Input Data
    N = len(x1)
    if not isinstance(x1, np.ndarray):
        x1 = np.array(x1)
    if x2 is not None:
        assert len(x2) == len(x1)
        if not isinstance(x2, np.ndarray):
            x2 = np.array(x2)
    ## Select Samples
    sample_size = int(N * sample_percent / 100)
    samples = np.random.choice(N, size=(samples, sample_size), replace=True)
    ## Compute Sample Measures
    values = []
    for sample_ind in tqdm(samples, desc="Bootstrap Sample", file=sys.stdout, total=len(samples)):
        if x2 is None:
            values.append(func(x1[sample_ind]))
        else:
            values.append(func(x1[sample_ind], x2[sample_ind]))
    ## Get Confidence Interval
    ci = np.percentile(values, q=[alpha*100/2, 50, 100-(alpha*100/2)])
    return ci

def summarize_performance(prediction_df):
    """
    Compute inference statistics on a DataFrame of predictions

    Args:
        prediction_df (pandas DataFrame): All predictions from k-fold results
    
    Returns:
        summary_df (pandas DataFrame): Classification stats for each group, fold
    """
    summary_df = []
    ## Cycle Through Group and Fold
    for group in ["train","dev"]:
        for fold in prediction_df["fold"].unique():
            ## Isolate Subset
            pred_subset = prediction_df.loc[(prediction_df["fold"]==fold)&
                                            (prediction_df["group"]==group)]
            ## Score Performance
            pred_performance = {
                "fold":fold,
                "group":group,
                "median_error":pred_subset["error"].median(),
                "mean_error":pred_subset["error"].mean(),
                "acc_at_100":(pred_subset["error"]<=100).sum() / len(pred_subset)
            }
            ## Reverse Geocode Coordinates
            true_discrete = reverse_search(pred_subset[["longitude_true","latitude_true"]].values)
            pred_discrete = reverse_search(pred_subset[["longitude_pred","latitude_pred"]].values)
            ## Classification Metrics (Various Levels)
            levels = ["name","admin1","cc"]
            level_names = ["city","state","country"]
            for l, (level, level_name) in enumerate(zip(levels, level_names)):
                l_true = list(map(lambda i: ", ".join([i[j] for j in levels[l:]]), true_discrete))
                l_pred = list(map(lambda i: ", ".join([i[j] for j in levels[l:]]), pred_discrete))
                for score_func, score_name in zip([metrics.f1_score, metrics.precision_score, metrics.recall_score, metrics.accuracy_score],
                                                  ["f1","precision","recall","accuracy"]):
                    if score_name == "accuracy":
                        score = score_func(l_true,
                                           l_pred,
                                           sample_weight=None)
                    else:
                        score = score_func(l_true,
                                           l_pred,
                                           average="weighted",
                                           zero_division=0)
                    pred_performance[f"{level}_{score_name}"] = score
            summary_df.append(pred_performance)
    ## Concatenate Results
    summary_df = pd.DataFrame(summary_df)
    ## Append Prediction Metadata
    non_meta_cols = ["longitude_true",
                     "longitude_pred",
                     "latitude_true",
                     "latitude_pred",
                     "error",
                     "num_comments",
                     "fold",
                     "group"]
    meta_dict = prediction_df.iloc[0].drop(non_meta_cols).to_dict()
    for col, val in meta_dict.items():
        summary_df[col] = val
    return summary_df

def accuracy_at_d(error, d = 100):
    """
    Compute inference accuracy at a mile threshold d

    Args:
        error (1d-array): Inference error in miles
        d (float): Max threshold to consider a correct inference
    
    Returns:
        acc (float): Percentage of correct samples within threshold distance
    """
    if not isinstance(error, np.ndarray):
        error = np.array(error)
    acc = (error <= d).sum() / len(error)
    return acc

def get_confidence_intervals(prediction_df,
                             n_samples=10,
                             sample_percent=30,
                             alpha=0.05):
    """
    Get confidence intervals around major classification statistics

    Args:
        prediction_df (pandas DataFrame): Cross validation predictions
        n_samples (int): Number of bootstrap samples
        sample_percent (float [0,100]): Percentage of available samples to consider per
                                        bootstrap sample
        alpha (float [0,0.5]): Type 1 error rate (e.g. 0.05 = 95% confidence level)
    """
    confidence_intervals = {"train":{}, "dev":{}}
    for group in ["train","dev"]:
        group_data = prediction_df.loc[prediction_df["group"]==group]
        y_true = group_data[["longitude_true","latitude_true"]].values
        y_pred = group_data[["longitude_pred","latitude_pred"]].values
        error = group_data["error"].values
        confidence_intervals[group]["median_error"] = bootstrap_confidence_interval(error,
                                                                                    func=np.median,
                                                                                    samples=n_samples,
                                                                                    sample_percent=sample_percent,
                                                                                    alpha=alpha)
        confidence_intervals[group]["mean_error"] = bootstrap_confidence_interval(error,
                                                                                  func=np.mean,
                                                                                  samples=n_samples,
                                                                                  sample_percent=sample_percent,
                                                                                  alpha=alpha)
        confidence_intervals[group]["acc_at_100"] = bootstrap_confidence_interval(error,
                                                                                  func=accuracy_at_d,
                                                                                  samples=n_samples,
                                                                                  sample_percent=sample_percent,
                                                                                  alpha=alpha)
        ## Reverse Geocode Coordinates
        true_discrete = reverse_search(y_true)
        pred_discrete = reverse_search(y_pred)
        ## Classification Metrics (Various Levels)
        levels = ["name","admin1","cc"]
        level_names = ["city","state","country"]
        for l, (level, level_name) in enumerate(zip(levels, level_names)):
            l_true = np.array(list(map(lambda i: ", ".join([i[j] for j in levels[l:]]), true_discrete)))
            l_pred = np.array(list(map(lambda i: ", ".join([i[j] for j in levels[l:]]), pred_discrete)))
            for score_func, score_name in zip([metrics.f1_score, metrics.precision_score, metrics.recall_score, metrics.accuracy_score],
                                                ["f1","precision","recall","accuracy"]):
                if score_name == "accuracy":
                    ci = bootstrap_confidence_interval(l_true,
                                                       l_pred,
                                                       func=lambda x, y: score_func(x, y),
                                                       samples=n_samples,
                                                       sample_percent=sample_percent,
                                                       alpha=alpha)
                else:
                    ci = bootstrap_confidence_interval(l_true,
                                                       l_pred,
                                                       func=lambda x, y: score_func(x, y, average="weighted", zero_division=0),
                                                       samples=n_samples,
                                                       sample_percent=sample_percent,
                                                       alpha=alpha)
                confidence_intervals[group][f"{level}_{score_name}"] = ci
    return confidence_intervals


def load_cross_validation_results(directory_name,
                                  n_samples=10,
                                  sample_percent=30,
                                  alpha=0.05):
    """
    Load results from a cross-validation run

    Args:
        directory (name): Name of the cross-validation run directory
        n_samples (int): Number of bootstrap samples to use for computing stats
        sample_percent (float [0,100]): Percentage of samples to user for each
                                        bootstrap sample
        alpha (float [0,0.5]): Type 1 error rate (e.g. 0.05 -> 95% Confidence Level)
    """
    full_path = f"{RESULTS_DIR}{directory_name}/"
    ## Load Configuration
    with open(f"{full_path}config.json","r") as the_file:
        config = json.load(the_file)
    ## Load Predictions From Each Fold
    fold_dirs = glob(f"{full_path}Fold_*/")
    prediction_df = []
    for fd in fold_dirs:
        ## Load Files
        train_predictions = pd.read_csv(f"{fd}train_predictions.csv", low_memory=False, index_col=0)
        dev_predictions = pd.read_csv(f"{fd}dev_predictions.csv", low_memory=False, index_col=0)
        ## Append Split Type
        train_predictions["group"] = "train"
        dev_predictions["group"] = "dev"
        ## Update Cache
        prediction_df.append(train_predictions)
        prediction_df.append(dev_predictions)
    ## Concatenate Predictions
    prediction_df = pd.concat(prediction_df)
    prediction_df = prediction_df.sort_values(["fold","group"])
    ## Append Data Parameters from Config
    prediction_df["data_resolution"] = config["data"]["MIN_RESOLUTION"]
    prediction_df["data_min_comments"] = config["data"]["MIN_COMMENTS"]
    prediction_df["data_vocabulary_params"] = json.dumps(config["data"]["VOCAB_PARAMETERS"])
    ## Append Model Parameters from Config
    for attribute, value in config["model"].items():
        prediction_df[f"model_{attribute}"] = value
    ## Append Source
    prediction_df["source"] = directory_name
    ## Summarize
    summary_df = summarize_performance(prediction_df)
    ## Get Confidence Intervals
    confidence_intervals = get_confidence_intervals(prediction_df,
                                                    n_samples,
                                                    sample_percent,
                                                    alpha)
    return summary_df, confidence_intervals, prediction_df

def plot_bar(summary_stats_df,
             confidence_intervals,
             metric):
    """
    Create a bar plot showing performance over cross-validation results

    Args:
        summary_stats_df (pandas DataFrame): Concatenated summary statistics for all runs
        confidence_intervals (dict): Confidence intervals for each run, group, statistic
        metric (str): Name of the metric to plot
    
    Returns:
        fig, axes (matplotlib): Training and Development performance comparison between runs
    """
    ## Initialize Figure
    fig, axes = plt.subplots(1, 2, figsize=(10,5.8), sharey=True)
    ## Plot Data
    for g, group in enumerate(["train","dev"]):
        cur_ind = 0
        xticks = []
        for i, (_, name) in enumerate(CV_DIRECTORIES):
            plot_data = summary_stats_df.loc[(summary_stats_df["name"]==name)&
                                             (summary_stats_df["group"]==group)]
            x = cur_ind * 0.3
            ci = confidence_intervals[name][group][metric]
            axes[g].bar(x=x,
                        height=ci[2] - ci[0],
                        bottom=ci[0],
                        color ="C0",
                        width = .25,
                        alpha =0.5)
            axes[g].scatter([x for _ in range(len(plot_data))],
                            plot_data[metric].values,
                            color = "C0",
                            s = 50,
                            alpha = .9)
            if ANNOTATE_PLOTS:
                axes[g].text(x + (.25/2),
                            ci[1], "{:.2f}".format(ci[1]),
                            ha="left",
                            va="center",
                            bbox=dict(facecolor='white', alpha=1))
            axes[g].bar(x, height=0, bottom=ci[1], edgecolor="black", width=.25)
            xticks.append(x)
            cur_ind += 1
        axes[g].set_xticks(xticks)
        axes[g].set_xticklabels([name for _, name in CV_DIRECTORIES])
        if g == 0:
            axes[g].set_ylabel(metric.replace("_", " ").title(), fontweight="bold")
        axes[g].set_title(group.title(), loc = "left", fontweight="bold")
    fig.tight_layout()
    return fig, axes

def compute_support_effect(prediction_df):
    """

    """
    LOGGER.info("Computing Performance ~ Comment Support")
    ## Get Thresholds
    max_thresh = prediction_df.groupby(["source","group","fold"])["num_comments"].max().min()
    thresholds = [0]
    while thresholds[-1] < max_thresh:
        thresholds.append(int(min(max_thresh, thresholds[-1]+20)))
    ## Add Reverse Geocoding
    prediction_df["reverse_true"] = reverse_search(prediction_df[["longitude_true","latitude_true"]].values)
    prediction_df["reverse_pred"] = reverse_search(prediction_df[["longitude_pred","latitude_pred"]].values)
    ## Compute Scores at Each Threshold
    support_effects = []
    for source in tqdm(prediction_df["source"].unique(), desc="Source", position=1, leave=False):
        for t in tqdm(thresholds, desc="Threshold", position=2, leave=False):
            ## Isolate Source/Threshold Subset
            source_data = prediction_df.loc[(prediction_df["source"]==source)&
                                            (prediction_df["num_comments"]>=t)]
            ## Cycle Through Group and Fold
            source_summary = []
            for group in ["train","dev"]:
                for fold in source_data["fold"].unique():
                    ## Isolate Subset
                    pred_subset = source_data.loc[(prediction_df["fold"]==fold)&
                                                    (prediction_df["group"]==group)]
                    ## Score Performance
                    pred_performance = {
                        "fold":fold,
                        "group":group,
                        "median_error":pred_subset["error"].median(),
                        "mean_error":pred_subset["error"].mean(),
                        "acc_at_100":(pred_subset["error"]<=100).sum() / len(pred_subset)
                    }
                    ## Classification Metrics (Various Levels)
                    levels = ["name","admin1","cc"]
                    level_names = ["city","state","country"]
                    for l, (level, level_name) in enumerate(zip(levels, level_names)):
                        l_true = list(map(lambda i: ", ".join([i[j] for j in levels[l:]]), pred_subset["reverse_true"].values))
                        l_pred = list(map(lambda i: ", ".join([i[j] for j in levels[l:]]), pred_subset["reverse_pred"].values))
                        for score_func, score_name in zip([metrics.f1_score, metrics.precision_score, metrics.recall_score, metrics.accuracy_score],
                                                          ["f1","precision","recall","accuracy"]):
                            if score_name == "accuracy":
                                score = score_func(l_true,
                                                   l_pred,
                                                   sample_weight=None)
                            else:
                                score = score_func(l_true,
                                                   l_pred,
                                                   average="weighted",
                                                   zero_division=0)
                            pred_performance[f"{level}_{score_name}"] = score
                    source_summary.append(pred_performance)
            source_summary = pd.DataFrame(source_summary)
            source_summary["score_threshold"] = t
            source_summary["num_users"] = len(source_data)
            source_summary["source"] = source
            support_effects.append(source_summary)
    ## Concatenate
    support_effects = pd.concat(support_effects).reset_index(drop=True)
    return support_effects

def plot_support_effect(support_effect_df,
                        metric):
    """

    """
    ## Aggregate
    scores_agg = support_effect_df.groupby(["group","source","score_threshold"]).agg({metric:[np.mean,np.std,len]})[metric]
    scores_agg = scores_agg.reset_index()
    ## Add Standard Error
    scores_agg["standard_error"] = scores_agg["std"] / np.sqrt(scores_agg["len"]-1)
    ## Plot
    fig, ax = plt.subplots(1, 2, figsize=(10,5.8), sharey=True)
    for g,group in enumerate(["train","dev"]):
        for s,(source,name) in enumerate(CV_DIRECTORIES):
            plot_data = scores_agg.loc[(scores_agg["source"]==source)&
                                       (scores_agg["group"]==group)]
            ax[g].errorbar(plot_data["score_threshold"],
                           plot_data["mean"],
                           plot_data["standard_error"],
                           color=f"C{s}",
                           label=name.replace("\n"," "))
        ax[g].set_xlabel("Comment Threshold",fontweight="bold")
        if g == 0:
            ax[g].set_ylabel(metric.replace("_", " ").title(), fontweight="bold")
        ax[g].legend(loc="upper right",frameon=True,fontsize=8)
        ax[g].set_title(group.title(),loc="left",fontweight="bold")
    fig.tight_layout()
    return fig, ax

def main():
    """
    Run quantitative analysis of inference performance, comparing
    multiple cross-validation runs

    Args:
        None

    Returns:
        None
    """
    ## Create Plot Directory
    LOGGER.info("Creating Analysis Directory")
    analysis_dir = f"{PLOTS_DIR}{ANALYSIS_NAME}/"
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    ## Get Results
    LOGGER.info("Loading and Summarizing Performance")
    summary_stats_df = []
    predictions_df = []
    confidence_intervals = {}
    for dir_name, dir_lbl in CV_DIRECTORIES:
        LOGGER.info(f"Processing Results For `{dir_lbl}`")
        model_summary, model_confidence_ints, model_preds = load_cross_validation_results(dir_name,
                                                                                          n_samples=CI_N_SAMPLES,
                                                                                          sample_percent=CI_SAMPLE_PERCENT,
                                                                                          alpha=CI_ALPHA)
        model_summary["name"] = dir_lbl
        summary_stats_df.append(model_summary)
        predictions_df.append(model_preds)
        confidence_intervals[dir_lbl] = model_confidence_ints
    summary_stats_df = pd.concat(summary_stats_df).reset_index(drop=True)
    predictions_df = pd.concat(predictions_df).reset_index(drop=True)
    ## Create Summary Figures
    plot_metrics = list(list(confidence_intervals.values())[0]["train"].keys())
    for m in plot_metrics:
        fig, ax = plot_bar(summary_stats_df, confidence_intervals, m)
        fig.savefig(f"{analysis_dir}{m}.png")
        plt.close(fig)
    ## Evaluation ~ Comment Support Thresholds
    support_effect_df = compute_support_effect(predictions_df)
    for m in plot_metrics:
        fig, ax = plot_support_effect(support_effect_df, m)
        fig.savefig(f"{analysis_dir}comment_support_{m}.png")
        plt.close(fig)

####################
#### Execute
####################

if __name__ == "__main__":
    _ = main()