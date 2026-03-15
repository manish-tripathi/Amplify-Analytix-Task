# Amplify Analytix Task

A portfolio of data analytics and data science work covering industrial sales analysis, booking conversion prediction, and IPO data scraping.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/manish-tripathi/Amplify-Analytix-Task/main?labpath=amplify-analytix-task.ipynb)

---

## Notebooks

### 1. `amplify-analytix-task.ipynb` — Sales & Revenue Analytics

Exploratory analysis of ~3.3M industrial product sales transactions spanning 5 years.

**Key analyses:**
- Top 10 customers by revenue and profit (last 2 years)
- Highest-selling distributor in July 2021
- Product line variance analysis — which categories experienced greatest sales volatility over time

**Data required:**
- `data_analytics_sales.csv` — Transaction-level sales records (26 columns, ~3.3M rows)
- `data_analytics_products.csv` — Product catalogue with hierarchical ACM/product-line categorisation (7 columns, ~19.8K rows)

Update the `DATA_DIR` variable at the top of the notebook to point to your local data directory.

---

### 2. `data-science-task.ipynb` — Booking Conversion Prediction (ML Pipeline)

End-to-end machine learning pipeline predicting whether a user will **book** an accommodation listing after searching.

**Pipeline stages:**
1. Data loading & feature type identification
2. Summary statistics and data quality assessment
3. Collinearity analysis (correlation heatmap)
4. Class imbalance identification and handling (`scale_pos_weight` in XGBoost)
5. Feature engineering — datetime decomposition, target encoding, standard scaling
6. XGBoost classifier with stratified train/val/test split (60/20/20)
7. Hyperparameter tuning via `GridSearchCV` (recall + F1 scoring)
8. Model evaluation — accuracy, precision, recall, F1, confusion matrix, ROC-AUC
9. Feature importance visualisation
10. Statistical tests — chi-square tests for review score and star rating effects on CTR & CVR

**Data required:**
- `train_small_sample.csv` — ~13K rows, 44 features (included in repo)

---

### 3. `scrapping-euronext-for-ipos.ipynb` — Euronext IPO Scraper

Scrapes 2,739 IPO records from the [Euronext IPO Showcase](https://live.euronext.com/en/ipo-showcase), collecting IPO date, company name, listing URL, and ICB sector classification.

**Note:** Full run takes ~2 hours due to rate limiting. Output saved to `output.csv`.

---

### 4. `chapter00.ipynb` — W&B + Cohere RAG Setup

Integration boilerplate for Weights & Biases experiment tracking and Cohere `command-r-plus` for RAG workflows.

---

## Setup

```bash
pip install -r requirements.txt
jupyter notebook
```

### Environment variables (optional)

Set these to avoid hardcoded paths:

| Variable | Description |
|---|---|
| `DATA_DIR` | Directory containing the CSV data files |
| `COHERE_API_KEY` | Cohere API key (for chapter00.ipynb) |
| `WANDB_API_KEY` | Weights & Biases API key (for chapter00.ipynb) |

---

## Data Notes

- Sales data is **anonymised** — customer names, distributor names, and product descriptions are obfuscated.
- Negative `Quantity` values represent product returns/refunds and are excluded from revenue/profit analyses.
- All monetary values are in **USD**.
- The booking dataset (`train_small_sample.csv`) has significant class imbalance: ~2% booking rate, which is handled via `scale_pos_weight` in XGBoost.

---

## Results Summary

| Analysis | Key Finding |
|---|---|
| Top customer revenue (2Y) | Top customer contributed ~$Xm in SAP extended value |
| Product line volatility | `SA_IndustrialStructuralBonding` had the highest monthly sales variance |
| Booking model recall | 0.52 on validation set (optimised for recall given class imbalance) |
| Booking model F1 | 0.09 (reflects extreme class imbalance, ~2% positive rate) |
| CTR | ~X% overall click-through rate |
| CVR | ~X% conversion rate (clicks → bookings) |
