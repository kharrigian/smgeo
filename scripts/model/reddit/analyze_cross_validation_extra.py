
## Reverse Geocode Coordinates
true_discrete = reverse_search(predictions_df[["longitude_true","latitude_true"]].values)
pred_discrete = reverse_search(predictions_df[["longitude_pred","latitude_pred"]].values)
## Classification Metrics (Various Levels)
levels = ["name","admin1","cc"]
level_names = ["city","state","country"]
for l, (level, level_name) in enumerate(zip(levels, level_names)):
    l_true = [i.get(level) for i in true_discrete]
    l_pred = [i.get(level) for i in pred_discrete]
#    l_true = list(map(lambda i: ", ".join([i[j] for j in levels[l:]]), true_discrete))
#    l_pred = list(map(lambda i: ", ".join([i[j] for j in levels[l:]]), pred_discrete))
    predictions_df[f"{level_name}_true"] = l_true
    predictions_df[f"{level_name}_pred"] = l_pred

dev_data = predictions_df.loc[predictions_df["group"] == "dev"].copy()
dev_data.loc[dev_data["country_true"]!="US", "state_true"] = "Non US"
dev_data.loc[dev_data["country_pred"]!="US", "state_pred"] = "Non US"
top_countries = set(dev_data["country_true"].value_counts().nlargest(50).index)
dev_data.loc[~dev_data["country_true"].isin(top_countries), "country_true"] = "Other"
dev_data.loc[~dev_data["country_pred"].isin(top_countries), "country_pred"] = "Other"
dev_data.rename(columns={"true_tz":"tz_true","pred_tz":"tz_pred"}, inplace=True)
timezones = set(sorted(dev_data.loc[dev_data["country_true"]=="US"]["tz_true"].unique()))
dev_data["tz_true"] = dev_data["tz_true"].map(lambda i: i if i in timezones else "Other")
dev_data["tz_pred"] = dev_data["tz_pred"].map(lambda i: i if i in timezones else "Other")


group = "tz"
ignore = set([])
min_comments = 100
suffix = "100_comment_min"


subset = dev_data.loc[dev_data["num_comments"]>=min_comments]
y_true = subset[f"{group}_true"].tolist()
y_pred = subset[f"{group}_pred"].tolist()
keepers = [i for i, y in enumerate(y_true) if y not in ignore]
y_true = [y_true[j] for j in keepers]
y_pred = [y_pred[j] for j in keepers]
labels = pd.Series(y_true).value_counts().sort_values(ascending=False).index.tolist()
argmax = pd.Series(y_true).value_counts().idxmax()
y_argmax = [argmax] * len(y_true)
if len(suffix) > 0:
    suffix = f"_{suffix}"


states_cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)
states_recall = metrics.recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
states_argmax_recall = metrics.recall_score(y_true, y_argmax, labels=labels, average="macro", zero_division=0)
states_precision = metrics.precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
states_argmax_precision = metrics.precision_score(y_true, y_argmax, labels=labels, average="macro", zero_division=0)
states_report = metrics.classification_report(y_true, y_pred, labels=labels, zero_division=0)
states_argmax_accuracy = metrics.accuracy_score(y_true, y_argmax)
states_accuracy = metrics.accuracy_score(y_true, y_pred)
states_error_report = subset.groupby([f"{group}_true"]).agg({"error":[len, np.mean, np.median, accuracy_at_d]})["error"]
states_error_report = states_error_report.loc[labels]
report_str = """
{}
{}
{}
Classification Report:
{}""".format(
"Recall: {:.4f} (Argmax [{}] Baseline = {:.4f})".format(states_recall, argmax, states_argmax_recall),
"Precision: {:.4f} (Argmax [{}] Baseline = {:.4f})".format(states_precision, argmax, states_argmax_precision),
"Accuracy: {:.4f} (Argmax [{}] Baseline = {:.4f})".format(states_accuracy, argmax, states_argmax_accuracy),
states_report)
print(report_str)


with open(f"{group}{suffix}.classification_report.txt","w") as the_file:
    the_file.write(report_str)
states_cm_df = pd.DataFrame(states_cm,index=labels,columns=labels)
states_cm_df.to_csv(f"{group}{suffix}.confusion.csv")
states_error_report.to_csv(f"{group}{suffix}.error_report.csv")

states_cm_row_norm = np.divide(states_cm.astype(float),
                               states_cm.sum(axis=1, keepdims=True).astype(float),
                               where=states_cm.sum(axis=1)>0,
                               out = np.zeros_like(states_cm).astype(float))
states_cm_col_norm = np.divide(states_cm.astype(float),
                               states_cm.sum(axis=0).astype(float),
                               where=states_cm.sum(axis=0)>0,
                               out = np.zeros_like(states_cm).astype(float))

fig, ax = plt.subplots(1,2,figsize=(14,7),sharey=True)
ax[0].imshow(states_cm_row_norm,
             cmap=plt.cm.Purples,
             aspect="auto",
             vmin=0,
             vmax=1)
m = ax[1].imshow(states_cm_col_norm,
             cmap=plt.cm.Purples,
             aspect="auto",
             vmin=0,
             vmax=1)
ax[0].set_yticks(list(range(len(labels))))
ax[0].set_yticklabels(labels, fontsize=6)
for i in range(2):
    ax[i].set_xticks(list(range(len(labels))))
    ax[i].set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax[i].set_xlabel("Predicted Label", fontweight="bold", fontsize=12)
ax[0].set_ylabel("True Label", fontweight="bold", fontsize=12)
ax[0].set_title("Recall (Proportion of True)")
ax[1].set_title("Precision (Proportion of Predicted)")
fig.colorbar(m)
fig.tight_layout()
fig.savefig(f"{group}{suffix}.confusion_matrices.png",dpi=300)
plt.close()
