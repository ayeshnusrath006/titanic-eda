"""
Task 2: Exploratory Data Analysis (EDA) - Titanic Dataset
Elevate Labs AI & ML Internship

Run with: python eda.py
Outputs summary stats to console and saves all plots to the plots/ folder.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
df = pd.read_csv("data/titanic.csv")

print("=" * 60)
print("SHAPE:", df.shape)
print("=" * 60)
print("\nCOLUMN INFO:")
print(df.info())

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

# ----------------------------------------------------------------------
# 2. Summary statistics
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY STATISTICS (numeric features)")
print("=" * 60)
summary = df.describe().T
summary["median"] = df.median(numeric_only=True)
summary["skew"] = df.skew(numeric_only=True)
print(summary)
summary.to_csv("plots/summary_statistics.csv")

print("\n" + "=" * 60)
print("SURVIVAL RATE BY CATEGORY")
print("=" * 60)
print("\nOverall survival rate: {:.2%}".format(df["Survived"].mean()))
print("\nBy Sex:\n", df.groupby("Sex")["Survived"].mean())
print("\nBy Pclass:\n", df.groupby("Pclass")["Survived"].mean())
print("\nBy Embarked:\n", df.groupby("Embarked")["Survived"].mean())

# ----------------------------------------------------------------------
# 3. Histograms for numeric features
# ----------------------------------------------------------------------
numeric_cols = ["Age", "Fare", "SibSp", "Parch"]
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.flatten(), numeric_cols):
    sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue")
    ax.set_title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig("plots/01_histograms.png")
plt.close()

# ----------------------------------------------------------------------
# 4. Boxplots for numeric features (outlier detection)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
sns.boxplot(y=df["Age"], ax=axes[0], color="lightgreen")
axes[0].set_title("Boxplot: Age")
sns.boxplot(y=df["Fare"], ax=axes[1], color="salmon")
axes[1].set_title("Boxplot: Fare")
plt.tight_layout()
plt.savefig("plots/02_boxplots.png")
plt.close()

# Boxplot of Fare/Age split by Survived (helps spot patterns)
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
sns.boxplot(x="Survived", y="Age", data=df, ax=axes[0], palette="Set2")
axes[0].set_title("Age vs Survival")
sns.boxplot(x="Survived", y="Fare", data=df, ax=axes[1], palette="Set2")
axes[1].set_title("Fare vs Survival")
plt.tight_layout()
plt.savefig("plots/03_boxplots_by_survival.png")
plt.close()

# ----------------------------------------------------------------------
# 5. Correlation matrix
# ----------------------------------------------------------------------
corr_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
corr = df[corr_cols].corr()
plt.figure(figsize=(7, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("plots/04_correlation_heatmap.png")
plt.close()

# ----------------------------------------------------------------------
# 6. Pairplot for feature relationships
# ----------------------------------------------------------------------
pair_df = df[corr_cols].dropna()
g = sns.pairplot(pair_df, hue="Survived", palette="husl", diag_kind="kde")
g.fig.suptitle("Pairplot of Key Features by Survival", y=1.02)
g.savefig("plots/05_pairplot.png")
plt.close()

# ----------------------------------------------------------------------
# 7. Categorical feature breakdowns
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.countplot(x="Pclass", hue="Survived", data=df, ax=axes[0], palette="Set1")
axes[0].set_title("Survival Count by Pclass")
sns.countplot(x="Sex", hue="Survived", data=df, ax=axes[1], palette="Set1")
axes[1].set_title("Survival Count by Sex")
sns.countplot(x="Embarked", hue="Survived", data=df, ax=axes[2], palette="Set1")
axes[2].set_title("Survival Count by Embarked")
plt.tight_layout()
plt.savefig("plots/06_categorical_survival.png")
plt.close()

print("\nAll plots saved to the plots/ folder.")
