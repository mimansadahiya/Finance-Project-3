# Project Summary: U.S. Corporate Default Prediction

This document provides a high-level summary of the tasks completed for the Credit Risk / Corporate Bankruptcy Prediction project. It is updated continuously as we implement new models and analysis.

---

## 1. Project Objective & Workflow

The goal of this project is to build a predictive model that identifies which U.S. public companies are likely to enter bankruptcy in the subsequent fiscal year. 

The dataset contains accounting and market data for **8,262 companies** across **78,682 firm-year observations** from **1999 to 2018**.

### The Modeling Strategy
Our credit risk modeling is structured in three phases:
1. **Altman-Style Benchmark:** An accounting-based scoring model using standard ratios.
2. **Logistic Regression:** A statistical approach interpreting how key financial drivers (profitability, leverage, size, liquidity) affect default probability.
3. **Machine Learning Models:** Advanced models (e.g., Random Forests, XGBoost, Neural Networks) to capture non-linear interactions.

---

## 2. Completed Tasks

### Phase 1: Data Acquisition & Preprocessing
* **Data Source:** Loaded the *U.S. Company Bankruptcy Prediction Dataset* from Kaggle using `mlcroissant` to fetch the metadata directly.
* **Variable Renaming:** Standardized the original column names (e.g., mapping `X1` to `current_assets`, `X15` to `retained_earnings`) to make the dataset intuitive.
* **Data Integrity Checks:** 
  * Checked for missing values (none present in the raw files).
  * Analyzed distributions and extreme values across all 18 raw financial variables.

### Phase 2: Exploratory Data Analysis & Preprocessing
* **Sample Split:** Set up a chronological split to prevent "look-ahead" bias (evaluating on future data):
  * **Training Set:** 1999–2011 (55,927 observations)
  * **Validation Set:** 2012–2014 (10,473 observations)
  * **Test Set:** 2015–2018 (12,282 observations)
* **Class Imbalance Analysis:** Identified that bankruptcy is a rare event. Out of 78,682 observations, only a small percentage are bankrupt (target distribution: 73,462 alive vs. 5,220 failed).
* **Winsorization:** Implemented outlier clipping at the 1st and 99th percentiles (bounds fitted only on training data) to prevent denominator distortions (e.g., extremely high leverage ratios from near-zero denominators) from skewing the statistical models.

### Phase 3: Feature Engineering
* Calculated **15 new financial ratios** representing the key dimensions of corporate financial health:
  * **Liquidity:** Current Ratio, Working Capital to Assets
  * **Profitability:** Return on Assets (ROA), EBITDA Margin, Gross Margin, EBIT Margin, Retained Earnings to Assets
  * **Leverage:** Debt Ratio, Long-Term Debt to Assets, Debt-to-Equity
  * **Efficiency:** Asset Turnover, Inventory Turnover, Receivables Turnover
  * **Market Valuation:** Market Value to Liabilities, Market Value to Assets
* **Handling Division by Zero:** Handled cases where denominator variables (like Total Liabilities or Inventory) were zero by replacing them with median values to avoid model distortion.

### Phase 4: Logistic Regression Modeling
We built and evaluated two logistic regression specifications in a dedicated Jupyter notebook (**[Logistic_Regression.ipynb](file:///Users/midahiya/Downloads/Fin%20Project%203/Logistic_Regression.ipynb)**):

#### A. Baseline Model (6 Motivated Variables)
Fitted an unweighted model using standard risk drivers:
* **Intercept**: `-2.4656` (represents log-odds of bankruptcy for a standard firm)
* **Coefficients & Interpretations**:
  * `debt_ratio` (Leverage): `+0.1908` (Higher debt ratio increases default odds by **21.0%** per standard deviation).
  * `market_to_liabilities` (Valuation): `-0.1332` (Stronger market value relative to liabilities decreases default odds by **12.5%** per standard deviation).
  * `roa` (Profitability): `-0.0765` (Higher profitability reduces default odds by **7.4%** per standard deviation).
  * `log_total_assets` (Size): `-0.0019` (Larger firms have slightly lower default odds, though effect size is small when controlling for leverage and profitability).
  * `working_capital_to_assets` (Liquidity) & `retained_earnings_to_assets` (Cumulative Profitability): Had slightly positive signs (`+0.0984` and `+0.0568`). While in isolation liquidity reduces default risk, in a multivariate model their signs flip due to **multicollinearity** (high correlation with `roa` and `debt_ratio`).

#### B. Penalized Model (LASSO Regularization)
Used L1 regularization across all 15 ratios, tuning the parameter $C$ on validation data (best $C = 100.0$).
* Regularization helped shrink less informative features (like asset turnover) while highlighting critical predictors like current liabilities, profit margins, and long-term debt.

#### C. Out-of-Sample Performance Comparison
Both models were evaluated on the chronological Test Set (2015–2018) with 287 actual bankruptcies:

| Metric | Baseline Model | LASSO Model |
| :--- | :---: | :---: |
| **Test ROC-AUC** | **0.7367** | 0.6992 |
| **Test PR-AUC** | 0.0555 | **0.0614** |
| **Top 5% Risk Bucket Recall** | 14.29% (41 defaults caught) | **21.60%** (62 defaults caught) |
| **Top 10% Risk Bucket Recall** | **31.71%** (91 defaults caught) | 31.36% (90 defaults caught) |
| **Top 20% Risk Bucket Recall** | **56.45%** (162 defaults caught) | 45.99% (132 defaults caught) |

* *Takeaway:* The baseline model shows the highest overall ranking power (ROC-AUC: 73.7%). While the LASSO model performs significantly better at isolating the extreme high-risk cases (capturing 21.6% of defaults in the top 5% risk bucket compared to only 14.3% in the baseline), the baseline model performs much better once the scope is widened to the top 20% (capturing 56.5% of all defaults vs. 46.0% for LASSO).

---

## 3. Next Steps (In Progress)

* **Altman Score Benchmark:** Implement an Altman Z-score proxy to serve as the baseline credit-risk benchmark.
* **Machine Learning Models:** Train and tune Random Forest and XGBoost classifiers.
* **Model Evaluation & Comparison:** Compare all models using ROC-AUC, PR-AUC, and Recall within the Top 5%, 10%, and 20% risk buckets.
