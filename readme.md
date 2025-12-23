# SBA Loan Default Risk Modeling — “Should This Loan Be Approved or Denied?”

This repo contains my end-to-end machine learning work for the **SBA National** small-business loan dataset, inspired by the case study:

> **Li, M., Mickel, A., & Taylor, S. (2018).** *“Should This Loan be Approved or Denied?”: A Large Dataset with Class Assignment Guidelines.* Journal of Statistics Education.

The original classroom case frames you as a **loan officer** who must decide whether to **approve or deny** a loan by estimating the **probability of default** (PD). The paper uses **logistic regression** as a baseline, and suggests extending the work with more advanced models.

In my notebook, I:
- Clean and feature-engineer the SBA dataset
- Train multiple classification models (baseline + advanced)
- Evaluate models using **standard metrics** (Accuracy, ROC-AUC, etc.)
- Translate predictions into a **business decision rule** using a **profit/cost matrix** and **profit-based gains curves**


---

## Project goals

1. **Predict loan outcome** using historical SBA-guaranteed loan data  
   - Target: `MIS_Status` (Paid in Full vs Charged-Off/Default)
2. **Turn model scores into actions** (approve/deny)  
   - Choose a **probability cutoff** that maximizes expected profit
3. **Compare models** beyond accuracy  
   - Use cost-sensitive evaluation + gains curves to recommend an approval strategy


---

## Dataset

**National SBA (SBA National Data)**  
- **Size:** 899,164 observations, 27 variables  
- **Time range:** 1987–2014  
- **Target label:** `MIS_Status`  
  - `P I F` = Paid in Full  
  - `CHGOFF` = Charged Off / Default

> Note: The raw dataset includes fields that should not be used in modeling (PII such as borrower name, plus columns that can leak outcomes). The notebook drops/avoids these when building the modeling table.


---

## What’s in the notebook

### 1) Data cleaning
Key cleaning steps include:
- Dropping leaky/sparse fields (e.g., `ChgOffDate` — only present for defaulted loans)
- Converting currency fields (e.g., `DisbursementGross`, `GrAppv`, `SBA_Appv`) to numeric
- Parsing dates and extracting useful time features where appropriate
- Cleaning and standardizing categorical fields like:
  - `NewExist`, `UrbanRural`, `LowDoc`, `RevLineCr`, `ApprovalFY`

### 2) Feature engineering
Examples of engineered features:
- **Industry:** `NAICS` → 2-digit sector grouping (more stable than full NAICS)
- **Location:** ZIP → **ZIP3** and/or ZIP3 frequency (reduces high-cardinality ZIPs)
- **Franchise:** `FranchiseCode` → `FranchiseFlag` (binary)
- **Geography stability:** grouping states into size buckets (`StateSizeGroup`)

### 3) Modeling
Models trained and compared:
- **k-Nearest Neighbors (kNN)**
- **Decision Tree**
- **Bagging (ensemble trees)**
- **Random Forest**
- **XGBoost**
- **Regularized Logistic Regression**
  - Ridge / Lasso / Elastic Net
- **Neural Network (MLP)**

### 4) Evaluation (metrics + business value)
In addition to standard classification metrics, the notebook evaluates each model using a **profit/cost matrix**:

- **Approve & Paid-in-Full:** +5% of loan amount  
- **Approve & Default:** −25% of loan amount  
- **Deny:** 0  

Then it:
- Sweeps cutoff thresholds from 0.01 → 0.99
- Picks the cutoff that **maximizes net profit**
- Builds a **profit-based gains curve** to answer:  
  **“What fraction of the safest loans should we approve?”**


---

## Results snapshot (from my notebook)

Ranked by **max net profit** under the profit/cost assumptions above:

| Model | Best cutoff | ROC-AUC | Accuracy | Max net profit | Approve % (gains-optimal) |
|---|---:|---:|---:|---:|---:|
| Bagging | 0.22 | 0.9733 | 0.9255 | $5.29B | 78.16% |
| XGBoost | 0.14 | 0.9758 | 0.9173 | $5.26B | 76.71% |
| Decision Tree | 0.45 | 0.9669 | 0.8979 | $5.18B | 74.19% |
| Random Forest | 0.18 | 0.9565 | 0.8921 | $4.84B | 75.13% |
| Neural Net (MLP) | 0.19 | 0.9091 | 0.8520 | $4.82B | 74.50% |
| kNN | 0.16 | 0.8032 | 0.7463 | $3.35B | 67.55% |
| Ridge Logistic | 0.58 | 0.8318 | 0.7813 | $1.38B | 69.53% |
| Lasso Logistic | 0.58 | 0.8319 | 0.7815 | $1.38B | 69.51% |
| Elastic Net Logistic | 0.58 | 0.8319 | 0.7816 | $1.38B | 69.49% |

**Takeaway:** In this setup, tree-based ensembles (Bagging / XGBoost) produced the strongest ROC-AUC and the highest profit under the assumed cost matrix.


