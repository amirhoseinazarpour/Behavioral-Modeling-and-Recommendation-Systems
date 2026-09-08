# Behavioral Modeling and Recommendation Systems

End-to-end exploratory data analysis, behavioral feature engineering, dimensionality reduction, and personalized product reordering prediction on the **Instacart Market Basket Analysis** dataset.

The project models grocery-shopping behavior at three complementary levels—**user**, **product**, and **user–product interaction**—then trains gradient-boosting models to estimate the probability that a user will reorder a product. Beyond binary classification, it evaluates the system as a recommender using user-level ranking metrics such as Precision@K, Recall@K, MAP, NDCG, Hit Rate, and MRR.

> **Primary artifact:** `Behavioral-Modeling-and-Recommendation-Systems.ipynb`

## Highlights

- Conducts comprehensive EDA of orders, basket sizes, reorder behavior, product popularity, department/aisle distributions, and temporal purchase patterns.
- Quantifies user–product interaction sparsity and investigates long-tail product demand.
- Builds behavioral features at user, product, and user–product levels.
- Applies PCA, t-SNE, and optionally UMAP to explore user behavior in reduced-dimensional spaces.
- Trains and tunes **LightGBM** and **XGBoost** models for product-reorder prediction using stratified cross-validation and ROC-AUC optimization.
- Compares models using classification metrics, confusion matrices, ROC curves, and recommendation-focused ranking metrics at K = 5 and K = 10.

## Problem Formulation

Given a user \(u\), a candidate product \(p\), and behavioral/contextual features \(\mathbf{x}_{u,p}\), the objective is to learn:

\[
\hat{y}_{u,p} = P(y_{u,p}=1 \mid \mathbf{x}_{u,p}),
\]

where \(y_{u,p}=1\) indicates that user \(u\) reorders product \(p\) in a future order.

For recommendation, candidate products are ranked by predicted reorder probability \(\hat{y}_{u,p}\), and the top-\(K\) products are returned for each user.

## Dataset

This notebook expects the original CSV files from the [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis/data) dataset:

| File | Description |
|---|---|
| `orders.csv` | Order-level metadata, including user identifier, order sequence, day/time, and prior-order interval |
| `order_products__prior.csv` | Products from users' prior orders, including reorder labels |
| `order_products__train.csv` | Products in held-out training orders |
| `products.csv` | Product names and category identifiers |
| `aisles.csv` | Aisle metadata |
| `departments.csv` | Department metadata |

Place these files in the same working directory as the notebook, or update the input paths in the data-loading cell.

> The Instacart dataset is large. Make sure that sufficient RAM and disk space are available before executing the full notebook.

## Pipeline

### 1. Data loading and validation

- Loads all six source tables.
- Inspects schema, missing values, duplicate order identifiers, and basic data quality issues.
- Merges prior-order, order, product, aisle, and department information for analysis.

### 2. Exploratory behavioral analysis

The notebook investigates:

- Dataset scale and user–product matrix sparsity.
- Orders per user and basket-size distributions.
- Global reorder rate and class imbalance.
- Product popularity and long-tail behavior.
- Department and aisle purchasing patterns.
- Order timing by day of week and hour of day.
- Missing values and basket-size outliers using the IQR rule.

Generated figures may include:

```text
eda_orders_per_user.png
eda_class_imbalance.png
eda_dept_aisle.png
eda_reorder_by_time.png
```

### 3. Feature engineering

The project derives three feature groups.

| Feature group | Examples |
|---|---|
| User-level | Total prior orders, mean/std of inter-order interval, average basket size, reorder ratio, days since last prior order |
| Product-level | Purchase count, global reorder rate, popularity percentile, purchase recency, unique-user count |
| User–product-level | Interaction count, last order number containing the product, pairwise reorder rate, user purchase share, orders since last purchase |

The engineered dataset is exported as:

```text
instacart_features_phase1.csv
```

### 4. Behavioral representation analysis

User-level features are standardized and analyzed with:

- **PCA** for linear compression, explained variance analysis, scree plots, and feature-loading interpretation.
- **t-SNE** for nonlinear visualization under several perplexity values.
- **UMAP** for manifold learning, if `umap-learn` is available.

### 5. Predictive modeling

Two gradient-boosted tree models are trained for binary reorder prediction:

| Model | Tuning approach | Main optimization target |
|---|---|---|
| LightGBM | Grid search with stratified 3-fold cross-validation | ROC-AUC |
| XGBoost | Grid search with stratified 3-fold cross-validation | ROC-AUC |

The notebook reports classification-oriented metrics:

- ROC-AUC
- Precision
- Recall
- F1-score
- Confusion matrix
- Specificity / True Negative Rate
- Negative Predictive Value (NPV)

### 6. Top-K recommendation evaluation

Since reorder prediction is ultimately used to rank candidate products for each user, the notebook additionally evaluates:

- Precision@5 and Precision@10
- Recall@5 and Recall@10
- Hit Rate@K
- MAP@K
- NDCG@K
- MRR@K

This evaluation separates probability calibration/classification performance from the quality of the resulting ranked recommendations.

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install numpy pandas matplotlib seaborn scikit-learn scipy
pip install lightgbm xgboost umap-learn jupyter
```

If you use Google Colab or a notebook environment, install any missing package in a separate cell and restart the kernel when required.

## Running the Notebook

```bash
jupyter lab
```

Then open:

```text
Behavioral-Modeling-and-Recommendation-Systems.ipynb
```

### Recommended execution order

1. Install dependencies, if necessary.
2. Download the Instacart CSV files and configure their paths.
3. Run the data-loading and validation cells.
4. Run EDA and feature engineering.
5. Confirm that `instacart_features_phase1.csv` is created.
6. Run PCA / t-SNE / UMAP analysis.
7. Train LightGBM and XGBoost.
8. Run classification and ranking evaluation cells.

## Repository Structure

```text
.
├── Behavioral-Modeling-and-Recommendation-Systems.ipynb
├── README.md
├── orders.csv
├── order_products__prior.csv
├── order_products__train.csv
├── products.csv
├── aisles.csv
├── departments.csv
├── instacart_features_phase1.csv          # generated
└── *.png                                  # generated figures
```

## Reproducibility Notes

- Several cells define a fixed `RANDOM_STATE` / seed for reproducible model splits and dimensionality-reduction experiments.
- Model scores can still vary slightly across library versions, hardware, multithreading settings, and sampling-dependent steps such as t-SNE.
- The notebook includes more than one exploratory workflow. Some cells assume variables produced by earlier cells; execute it in order from top to bottom.
- UMAP is optional. The notebook contains fallback behavior when the package is unavailable.
- The default pipeline can be memory-intensive because it merges transaction-level data and creates user–product features.

## Limitations

- Candidate generation is based primarily on historically observed user–product interactions and sampled negatives; this does not fully solve large-scale catalog retrieval or cold-start recommendation.
- The binary target is strongly imbalanced, so ROC-AUC should be interpreted alongside Precision, Recall, F1, and ranking metrics.
- Reorder prediction is not equivalent to next-basket recommendation in every setting: availability, substitution, price, promotions, and session-level context are not modeled explicitly.
- Hyperparameter grids can be expensive. Start with a smaller sample or reduced search space for a quick smoke test.

## Technologies

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `LightGBM` · `XGBoost` · `Matplotlib` · `Seaborn` · `PCA` · `t-SNE` · `UMAP` · `Jupyter`

## License and Data Terms

This repository contains code and analysis only. The Instacart dataset is not redistributed; obtain it directly from Kaggle and comply with the dataset's applicable terms of use.
