# Instamart Customer Intelligence & Support Copilot

## Overview

An offline AI/ML project for a quick-commerce store (synthetic data, no API key, no internet needed).

| Part | What it does | Code |
|---|---|---|
| **A** | Cleans raw data, loads SQLite, runs 7 SQL queries | `src/data_pipeline.py`, `src/sql_analysis.py`, `sql/queries.sql` |
| **B** | Groups customers (K-means) and predicts who will stop ordering | `src/customer_models.py` |
| **C** | Sorts support tickets safely (hides personal data, retries, budgets, routing) | `src/triage.py` |
| **D** | Answers policy questions (RAG) and runs a safe refund agent | `src/policy_rag.py`, `src/refund_agent.py`, `src/run_scenarios.py` |
| **Web app** | Customer shop + help chat, and an admin console with an **Analytics** section for Parts A–D | `webapp/` |

`generate_data.py` and `mock_llm.py` are the supplied files and are **not modified**. Tech: Python 3.11+, pandas, numpy, scikit-learn, joblib, SQLite, Flask.

## Reproduce

```bash
python -m venv .venv
.venv\Scripts\activate              # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

python run_project.py               # runs everything (Parts A-D) in order
python -m pytest tests -q           # 103 tests
```

**Web app (optional):**

```bash
python -m webapp.manage restore     # first time: create the login database
python -m webapp.manage reset-password admin    # prints a new admin password
python -m webapp.app                # open http://127.0.0.1:5000/login
```

Log in as `admin` and open **Analytics** to see: business overview, SQL results, data-quality checks, segments and lapse model, ticket triage, policy Q&A, and the refund agent (run, crash, resume, ledger). Customers (`user01`–`user10`) can shop and use the 💬 Help chat.

**Main outputs:** `data/clean/*.csv`, `data/instamart.db`, `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, `outputs/retention_list.csv`, `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv`, `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `state/` (agent checkpoints).

## Part A

**Data foundation.** Raw data: 1,200 customers, 40 products, 13,084 order rows, 164 tickets.

| Problem found in raw orders | Count | Fix |
|---|---|---|
| Exact duplicate rows | 119 | removed |
| Old "pending" rows replaced by a later row | 656 | kept the latest row per order |
| Bad amounts (`N/A`, `error`, blank) | 120 | 112 recomputed from price × qty × discount |
| Negative delivery times | 106 | 105 set to missing |
| Missing delivery on completed orders | 774 | filled with category median |
| Missing delivery on cancelled orders | 1,199 | left empty (never delivered) |
| Customers: blank age / impossible age / blank gender | 61 / 6 / 30 | 67 ages filled with city median; gender → male/female/unknown |
| City spellings / gender spellings | 24 → 6 / 10 → 3 | standardised |

**Clean result:** 1,200 customers · 12,309 orders (11,110 completed, 1,199 cancelled) · revenue **₹2,998,668.80**. All **12/12** validation checks pass (`ALL CHECKPOINTS PASSED`).

**SQL queries** (`sql/queries.sql`, each checked against pandas — 8/8 match):

| Q1 big spenders (WHERE+HAVING) | Q2 revenue city×category (JOIN) | Q3 above-avg price | Q4 no completed order (LEFT JOIN) | Q5 above-avg spenders (CTE) | Q6 city ranks (window) | Q7 running revenue |
|---|---|---|---|---|---|---|
| 30 rows | 30 rows | 11 rows | 25 rows | 550 rows | 1,175 rows | 183 rows |

**Quick insights:** average order ₹269.91 vs median ₹135.00 (skewed, skewness 4.61); 897 high-value outliers kept; Hyderabad earns most (₹528,590), Chennai least (₹463,629); Electronics = 41% of revenue; no strong correlations (|r| ≤ 0.10).

## Part B

**Customer segments** (RFM + K-means, 1,175 customers; K=4 has the best silhouette 0.526):

| Segment | Customers | Recency (days) | Orders | Spend (₹) | Action |
|---|---|---|---|---|---|
| Premium | 138 | 24.1 | 6.8 | 7,130 | VIP support, retention offers |
| Loyal Frequent | 419 | 8.8 | 19.4 | 3,410 | Loyalty rewards, bundles |
| Casual | 436 | 28.9 | 3.8 | 1,058 | Recommendations |
| At Risk / Dormant | 182 | 109.0 | 2.2 | 686 | Win-back offer |

Silhouette for K=2…8: 0.465, 0.445, **0.526**, 0.480, 0.482, 0.416, 0.412. PCA: 2 components explain **85.2%** (63.4% + 21.8%).

**Lapse prediction** (will a customer skip September?): 1,156 customers, 443 lapsed (38.3%), split 924 train / 232 test. **No leakage:** features use only orders before 1 Sep 2025, preprocessing sits inside the pipeline, and the model is chosen by cross-validation.

| Model (GridSearchCV, F1) | CV F1 | Precision | Recall | **Test F1** |
|---|---|---|---|---|
| Logistic regression (C=100) | 0.651 | 0.644 | 0.652 | 0.648 |
| **Random forest (depth 8, leaf 10) — selected** | 0.687 | 0.633 | 0.640 | **0.637** ✅ (≥ 0.55) |
| Baseline "everyone lapses" | – | 0.384 | 1.000 | 0.555 |
| Logistic / forest with oversampling (imbalance fix) | – | 0.589 / 0.558 | 0.854 / 0.809 | 0.697 / 0.661 |

Most important features: spend (0.163) and order count (0.141); demographics add almost nothing. The model is saved with joblib and reloads with identical predictions. `retention_list.csv` = 10 Premium customers most likely to lapse (illustrative; 1 actually lapsed).

## Part C

**Ticket triage** on 164 tickets (billing / delivery / technical / fraud) using `mock_llm.py`:

| Result | Value |
|---|---|
| Classified from JSON / from keyword / invalid → human | 135 / 12 / 9 |
| Empty / too long (never sent to the model) | 4 / 4 |
| Temporary failures fixed by retry (exponential backoff + jitter) | 19 of 19 |
| Accuracy: few-shot vs zero-shot | **94.2%** vs 80.8% |
| Tokens: few-shot vs zero-shot | 12,520 vs 7,372 (≈ $0.0063) |
| Personal data hidden before any model call | 55 emails, 91 phones, 132 order IDs |
| Budget demo (1,000-token cap) | 11 classified, 145 stopped and sent to a human |

**Routing:** problem/invalid → human review (17) · fraud → security team (34) · Premium → premium desk (11) · others → support queue (102).

## Part D

**Policy assistant (RAG):** 5 policy files → 15 chunks (40 words, 10 overlap), TF-IDF + cosine similarity. Correct document found **12/12**, correct fact in answer **12/12**, off-topic questions correctly answered "not found" **2/2**. Without overlap only 10/12 facts are found.

**Refund agent** (ReAct loop with 5 tools). Rule: a completed order delivered in **more than 45 minutes** gets a full refund; up to ₹1,000 is automatic, above goes to a human.

| Scenario | Expected → Result | Model calls |
|---|---|---|
| S1 late order ₹128.25 | REFUND ₹128.25 ✅ | 5 |
| S2 late order ₹135.00 | REFUND ₹135.00 ✅ | 5 |
| S3 late order ₹1,999.00 | ESCALATE to human ✅ | 5 |
| S4 cancelled order | DENIED ✅ | 3 |
| S5 on-time delivery | DENIED ✅ | 4 |
| S6 unknown order | NOT_FOUND ✅ | 2 |
| S7 agent loops | stopped at 6 iterations ✅ | 6 |

**Safety checks:** token limit 200 → stopped at 354 tokens, no refund · timeout 0 → stopped before any call · crash after step 2 → resumed from checkpoint, refund ₹128.25 paid **exactly once** · crash after payment → duplicate blocked. Every run is logged in `agent_runs.jsonl` and every payment in `refund_ledger.jsonl`. All 11 checks pass.

<!-- react-start -->
**ReAct vs plan-execute.** ReAct decides each step after seeing the last result, which suits refunds because the right path depends on the order (missing, cancelled, on time, above ₹1,000). It is easy to audit and checkpoint, but uses more model calls, so it is capped at 6 iterations plus token and time limits. Plan-execute plans everything first: cheaper and predictable, but it breaks when an early step changes the situation. For a small, branching, money-moving task, ReAct with hard limits is the better fit.
<!-- react-end -->

## Limitations and Next Steps

- The original brief file was not available, so some figures (audit counts, RFM stats) are this project's own results with seed 42, not brief-checked values.
- Data is synthetic and the LLM is a rule-based mock, so accuracy says little about a real model.
- The lapse model beats "flag everyone" only modestly (F1 0.637 vs 0.555); the retention list uses a whole-period Premium label.
- The refund ledger is a file, not a real payment system; the web app runs on a development server.
- **Next:** tune for recall, plug in a real LLM behind the same interface, add an approval queue for escalated refunds, and deploy with HTTPS.
