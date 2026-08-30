# Task 2 — Exploratory Data Analysis (EDA)
**Elevate Labs — AI & ML Internship**

## Objective
Understand the Titanic dataset using descriptive statistics and visualizations, and draw basic feature-level inferences.

## Tools Used
- Python
- Pandas
- Matplotlib
- Seaborn

## Dataset
Titanic dataset (891 passengers, 12 columns) — `data/titanic.csv`.

## Project Structure
```
titanic-eda/
├── data/
│   └── titanic.csv
├── plots/
│   ├── 01_histograms.png
│   ├── 02_boxplots.png
│   ├── 03_boxplots_by_survival.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_pairplot.png
│   ├── 06_categorical_survival.png
│   └── summary_statistics.csv
├── eda.py
└── README.md
```

## How to Run
```bash
pip install pandas matplotlib seaborn
python eda.py
```
This prints summary statistics to the console and saves all charts to `plots/`.

## Key Findings

**Missing data**
- `Age` has 177 missing values (~20%), `Cabin` has 687 missing (~77%), `Embarked` has 2 missing.

**Summary statistics**
- Overall survival rate: **38.4%**.
- Average age ≈ 29.7 years (median 28); average fare ≈ $32.20 (median $14.45) — the large gap between mean and median fare signals strong right-skew.

**Distributions (histograms)**
- `Age` is roughly bell-shaped with a slight right skew and a spike of infants/young children.
- `Fare` is heavily right-skewed — most tickets are cheap, with a long tail of very expensive first-class fares.
- `SibSp` and `Parch` are both right-skewed — most passengers travelled alone or with one family member.

**Outliers (boxplots)**
- `Fare` has many high-end outliers (up to $512), consistent with a few luxury first-class tickets.
- `Age` has a handful of high outliers (elderly passengers) but is otherwise fairly contained.
- Survivors have a noticeably higher median fare than non-survivors.

**Correlation matrix**
- `Fare` and `Pclass` are strongly negatively correlated (**-0.55**) — higher fares correspond to better (numerically lower) class.
- `Pclass` correlates negatively with `Survived` (**-0.34**) — poorer classes survived less often.
- `SibSp` and `Parch` are positively correlated (**0.41**) — passengers with siblings/spouses aboard often also had parents/children aboard (family groups).
- `Age` correlates negatively with `Pclass` (**-0.37**) — 1st-class passengers tended to be older.

**Categorical patterns**
- **Sex** is the single strongest survival signal: women survived **74.2%** of the time vs. men **18.9%**.
- **Pclass**: 1st class survival 63.0%, 2nd class 47.3%, 3rd class 24.2% — a clear class gradient.
- **Embarked**: passengers boarding at Cherbourg (C) had the highest survival rate (55.4%), likely because a larger share were 1st-class.

**Feature-level inferences**
- "Women and children first" plus class privilege both show up clearly in the data — sex and class are the two dominant predictors of survival.
- Fare is essentially a proxy for class/wealth and should not be used alongside `Pclass` in a model without care (multicollinearity risk).

---

## Interview Questions & Answers

**1. What is the purpose of EDA?**
EDA is the process of summarizing and visualizing a dataset before modeling, to understand its structure, distributions, relationships, missing data, and anomalies. It guides decisions on cleaning, feature engineering, and which modeling approach is appropriate.

**2. How do boxplots help in understanding a dataset?**
A boxplot shows the median, quartiles (IQR), and whiskers of a distribution, making it easy to spot skewness, spread, and outliers (points beyond 1.5×IQR from the quartiles) at a glance — much faster than scanning raw numbers.

**3. What is correlation and why is it useful?**
Correlation measures the strength and direction of the linear relationship between two numeric variables (from -1 to +1). It's useful for spotting which features move together, identifying potential predictors of a target variable, and flagging redundant/multicollinear features.

**4. How do you detect skewness in data?**
Visually, via histograms/density plots (a long tail on one side) or boxplots (median off-center, one whisker longer). Numerically, via the skewness statistic — values far from 0 indicate skew (positive = right tail, negative = left tail), and a large gap between mean and median is another quick signal.

**5. What is multicollinearity?**
It's when two or more predictor variables are highly correlated with each other (not just with the target), so they carry overlapping information. This can destabilize coefficient estimates in linear models and make it hard to isolate each feature's individual effect (e.g., `Fare` and `Pclass` here).

**6. What tools do you use for EDA?**
Pandas for data manipulation and summary statistics; Matplotlib and Seaborn for static visualizations (histograms, boxplots, heatmaps, pairplots); Plotly for interactive charts; sometimes Sweetviz/Pandas-Profiling for automated EDA reports.

**7. Can you explain a time when EDA helped you find a problem?**
In this task, EDA revealed that `Fare` had extreme outliers and a mean far above its median — a red flag that using the raw mean fare (rather than median, or a log transform) would misrepresent a "typical" passenger and could distort a downstream model.

**8. What is the role of visualization in ML?**
Visualization makes patterns, relationships, and data quality issues (outliers, skew, missing values, class imbalance) immediately visible in a way raw tables can't, which shapes preprocessing and feature engineering decisions and also helps communicate model behavior and results to others.

---
*Submitted as part of the Elevate Labs AI & ML Internship (MSME, Govt. of India).*
