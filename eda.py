"""data analysis. Saves all charts to outputs."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from features import add_features, load_data

os.makedirs("outputs", exist_ok=True)
sns.set_theme(style="whitegrid")
PALETTE = {"No Churn": "#3b6fb6", "Churn": "#d9534f"}

df = load_data()
df = df.drop(columns="CustomerID")        
df["churn_label"] = df["churn"].astype(str).str.lower().map(
    {"true": "Churn", "false": "No Churn", "yes": "Churn", "no": "No Churn"})
df["churn_num"] = (df["churn_label"] == "Churn").astype(int)


def save(name):
    plt.tight_layout()
    plt.savefig(f"outputs/{name}", dpi=150)
    plt.close()

plt.figure(figsize=(5, 4))
ax = sns.countplot(x="churn_label", data=df, order=["No Churn", "Churn"], palette=PALETTE,
                   hue="churn_label", legend=False)
for bar in ax.patches:
    ax.annotate(f"{int(bar.get_height())}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom")
plt.title(f"Churn Distribution ({df['churn_num'].mean():.1%} churn)")
plt.xlabel(""); plt.ylabel("Customers")
save("churn_distribution.png")

def churn_rate_by(column, title, filename):
    rate = df.groupby(column)["churn_num"].mean() * 100
    plt.figure(figsize=(5, 4))
    ax = sns.barplot(x=rate.index.astype(str), y=rate.values, color="#3b6fb6")
    for bar in ax.patches:
        ax.annotate(f"{bar.get_height():.1f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom")
    plt.title(title); plt.xlabel(column); plt.ylabel("Churn rate (%)")
    save(filename)


churn_rate_by("internationalplan", "Churn Rate vs International Plan", "churn_vs_international_plan.png")
churn_rate_by("voicemailplan", "Churn Rate vs Voicemail Plan", "churn_vs_voicemail_plan.png")

churn_rate_by("numbercustomerservicecalls", "Churn Rate vs Customer Service Calls",
              "churn_vs_service_calls.png")
usage = df.join(add_features(df.drop(columns=["churn", "churn_label", "churn_num"]))[["total_charge"]])
cols = ["totaldayminutes", "totaleveminutes", "totalnightminutes",
        "totalintlminutes", "total_charge", "numbervmailmessages"]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, col in zip(axes.ravel(), cols):
    sns.boxplot(x="churn_label", y=col, data=usage, order=["No Churn", "Churn"],
                palette=PALETTE, hue="churn_label", legend=False, ax=ax)
    ax.set_title(col); ax.set_xlabel(""); ax.set_ylabel("")
fig.suptitle("Usage vs Churn")
save("usage_vs_churn.png")

corr_df = df.drop(columns=["churn", "churn_label"]).copy()
for col in ["internationalplan", "voicemailplan"]:
    corr_df[col] = corr_df[col].astype(str).str.lower().isin(["true", "yes", "1"]).astype(int)
plt.figure(figsize=(12, 9))
sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0,
            annot_kws={"size": 7}, linewidths=0.3)
plt.title("Correlation Heatmap (charge and minutes columns are almost identical)")
save("correlation_heatmap.png")
