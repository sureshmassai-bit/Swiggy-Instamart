# Instamart Customer Intelligence & Support Copilot

## Overview

IIT Patna AI/ML capstone. A fully offline pipeline over **synthetic** quick-commerce data (no real customers, no API key, no network call):

| Part | What it does | Main code |
|---|---|---|
| A | Audit and clean raw CSVs, load SQLite, run 7 SQL queries and a pandas analysis | `src/data_pipeline.py`, `src/sql_analysis.py`, `sql/queries.sql` |
| B | RFM segmentation (K-means, PCA) and a leakage-safe lapse-prediction model | `src/customer_models.py` |
| C | Support-ticket triage with PII masking, retries, budgets and routing | `src/triage.py` |
| D | Policy RAG assistant and a checkpointed, resumable refund agent | `src/policy_rag.py`, `src/refund_agent.py`, `src/run_scenarios.py` |
| Web | Customer portal (order, raise tickets) and admin console (track customers, answer tickets) | `webapp/` |

`generate_data.py` and `mock_llm.py` are the supplied files and are used **unmodified**.

**Important about "reference checkpoints".** The original capstone brief document was not available in the workspace when this was implemented. Reference figures given in the task prompt (164 tickets; 135 / 12 / 9 statuses; 19 recovered failures; 15 chunks split 4/3/3/3/2; 12/12 documents and facts; S1-S7 outcomes; lapsed-class F1 >= 0.55) are checked and reported below. Reference figures that appear only in the brief (raw-audit numbers, cleaned row counts, RFM statistics, exact parameter grids and scoring) could **not** be verified; the values shown for those are what this implementation observed with seed 42, and tests pin them as regression checks, not as brief-verified facts.

## Reproduce

Python 3.11+ (developed on 3.13). Run every command from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

python run_project.py                # everything below, in dependency order
python -m pytest tests -q            # test suite
python tests/run_notebook.py notebooks/Customer_Intelligence_Support_Copilot.ipynb   # notebook top-to-bottom check
```

Or run the stages one by one (each needs the ones above it):

```bash
python generate_data.py              # data/raw/*.csv and data/policies/*.txt (seed 42)
python -m src.data_pipeline          # Part A: audit, clean, SQLite, checkpoints -> prints ALL CHECKPOINTS PASSED
python -m src.sql_analysis           # Part A: 7 SQL queries + pandas analysis + EDA notes
python -m src.customer_models        # Part B: segments, lapse model, retention list
python -m src.triage                 # Part C: triage (needs Part B's customer_segments.csv)
python -m src.policy_rag             # Part D: RAG evaluation + overlap comparison
python -m src.run_scenarios          # Part D: S1-S7, limits, crash/resume demos
```

The notebook `notebooks/Customer_Intelligence_Support_Copilot.ipynb` calls the same modules; open it and choose *Restart & Run All*. It is stored without outputs.

**Generated artifacts** (all reproducible): `data/clean/*.csv`, `data/instamart.db` (git-ignored), and in `outputs/`: `customer_segments.csv`, `lapse_model.joblib`, `retention_list.csv`, `triage_results.csv`, plus `audit_report.json`, `cleaning_report.json`, `eda_notes.md`, `sql_results/Q1-Q7.csv`, `kmeans_diagnostics.csv`, `segment_profile.csv`, `rfm_summary.csv`, `pca_segments.png`, `lapse_metrics.json`, `triage_zero_shot.csv`, `triage_summary.json`, `rag_evaluation.csv`, `rag_summary.json`, `rag_zero_overlap_failures.csv`, `agent_scenarios.csv`, `agent_demos.json`, `refund_ledger.jsonl`. Agent checkpoints go to `state/` (git-ignored).

## Part A — Data Foundation

**Raw audit** (`outputs/audit_report.json`, blanks counted as missing): customers 1200 x 5, products 40 x 4, orders 13,084 x 9, tickets 164 x 5.

| Raw issue | Count |
|---|---|
| Exact duplicate order rows | 119 |
| Pending rows later superseded by a final row | 656 |
| Distinct `order_id` | 12,309 |
| Non-numeric or blank `order_amount` (`N/A`, `error`) | 120 |
| Negative `delivery_minutes` | 106 |
| Blank `delivery_minutes` (1,211 on cancelled, 674 on completed, rest on dropped duplicates) | 2,541 |
| Customers: blank age / age outside 10-100 / blank gender | 61 / 6 / 30 |
| City spellings 24 -> 6; gender spellings 10 -> 3 | |

**Cleaning rules and rows changed** (`outputs/cleaning_report.json`):

| Rule | Rows |
|---|---|
| City `strip` + `title` | 476 changed |
| Gender -> male / female / unknown | 950 changed |
| Impossible ages (outside 10-100) -> missing; 67 missing ages then filled with the **city median** via `groupby().transform("median")` | 6 set missing |
| `order_ts` parsed; exact duplicates removed | 119 removed |
| Keep latest record per `order_id` | 656 removed |
| Invalid `order_amount` filled with price x quantity x (1 - discount/100) | 112 filled |
| Negative delivery -> missing | 105 set missing |
| **Completed** orders: missing delivery filled with the product-category median | 774 filled |
| **Cancelled** orders: delivery left missing (never delivered, so nothing to impute) | 1,199 kept missing |

Result: customers 1,200; orders 12,309 (11,110 completed, 1,199 cancelled); completed revenue INR 2,998,668.80. Tables are created from explicit DDL with primary keys, foreign keys and `CHECK` constraints; `order_ts` is stored as `YYYY-MM-DD HH:MM:SS`.

**Validation:** `python -m src.data_pipeline` runs 12 named checkpoints (tables, row counts, status split, revenue, null handling, keys, timestamp format, city/gender/age) and prints exactly `ALL CHECKPOINTS PASSED` only if all pass. Their expected values are observed values for this repo, not brief figures.

**SQL** (`sql/queries.sql`, run with `pandas.read_sql_query`, each result cross-checked against independent pandas code):

| Query | Pattern | Rows |
|---|---|---|
| Q1 | `WHERE` + `HAVING` | 30 |
| Q2 | three-table `JOIN`, revenue by city x category | 30 |
| Q3 | scalar subquery (price above average) | 11 |
| Q4 | `LEFT JOIN` (customers with no completed order) | 25 |
| Q5 | CTEs (above-average spenders) | 550 |
| Q6 | `RANK` / `DENSE_RANK` / `ROW_NUMBER` per city | 1,175 |
| Q7 | running total window | 183 |

**Pandas analysis:** Q2 is rebuilt with `pd.pivot_table(..., margins=True)` and equals the SQL result. Completed-order `order_amount`: mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, IQR 232.20, fences [-283.50, 645.30], 897 high outliers (kept: genuine high-value purchases). City totals range from INR 463,629 (Chennai) to 528,590 (Hyderabad); Electronics is 41% of revenue. Correlations among quantity, discount, amount and delivery time are all weak (|r| <= 0.10). The four-question EDA (about 110 words) is in `outputs/eda_notes.md`. **Correlation does not establish causation.**

## Part B — Customer Segments and Lapse Prediction

**RFM** (snapshot 2025-10-01; customers with >= 1 completed order): 1,175 customers. Recency mean 33.6 days, frequency mean 9.5, monetary mean INR 2,552 (`outputs/rfm_summary.csv`).

**K-means** (`StandardScaler`, `n_init=10`, `random_state=42`):

| K | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| Inertia | 1820.8 | 1268.2 | 743.1 | 617.4 | 503.2 | 408.5 | 366.5 |
| Silhouette | 0.465 | 0.445 | **0.526** | 0.480 | 0.482 | 0.416 | 0.412 |

**K = 4** is chosen from {3, 4, 5}: it has the best silhouette overall, the elbow flattens after it (K=3 -> 4 removes 41% of inertia, K=4 -> 5 only 17%), and four segments map to four distinct business actions. Cluster ids are labels, not rankings; names come from each cluster's profile.

| Segment | Size | Avg recency (days) | Avg frequency | Avg monetary (INR) | Action |
|---|---|---|---|---|---|
| **Premium** (highest monetary, `is_premium`) | 138 | 24.1 | 6.8 | 7,130 | VIP support, early access, proactive retention offers |
| Loyal Frequent | 419 | 8.8 | 19.4 | 3,410 | Loyalty rewards, bundles, nudge to bigger baskets |
| Casual | 436 | 28.9 | 3.8 | 1,058 | Recommendations, first-repeat incentives |
| At Risk / Dormant | 182 | 109.0 | 2.2 | 686 | Win-back offer before they churn |

**PCA** (2 components on scaled RFM): explained variance 63.4% + 21.8% = **85.2%** (plot: `outputs/pca_segments.png`). `outputs/customer_segments.csv` has the required columns `customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium`.

**Lapse model.** Cohort: customers with a completed order before **2025-09-01** (1,156; 443 lapsed = 38.3%). Label: no completed order in September 2025. Split: `train_test_split(test_size=0.2, stratify=y, random_state=42)` -> 924 train / 232 test. The 11 features are recency, frequency, monetary, average discount, cancellation rate, average delivery time, favourite category, city, age, gender and tenure.

*Leakage controls:* every feature is computed from orders strictly before 2025-09-01 (tested by tampering with all later orders: features do not change); no segment label or target is a feature; the median imputer, scaler and one-hot encoder live inside a `Pipeline` and are fitted on training folds only; no feature column has missing values (asserted); the model is chosen by **5-fold CV on the training set**, never by test results.

| Model (GridSearchCV, 5-fold, scoring = F1) | Best params | CV F1 | Test precision | Test recall | Test F1 | Confusion matrix `[[TN,FP],[FN,TP]]` |
|---|---|---|---|---|---|---|
| Logistic regression, C in {0.01, 0.1, 1, 10, 100} | C=100 | 0.651 | 0.644 | 0.652 | 0.648 | [[111, 32], [31, 58]] |
| **Random forest** (100 trees), depth {3,5,8} x leaf {5,10,20} | depth 8, leaf 10 | **0.687** | 0.633 | 0.640 | **0.637** | [[110, 33], [32, 57]] |
| Baseline: never lapsed | | | 0.000 | 0.000 | 0.000 | [[143, 0], [89, 0]] |
| Baseline: always lapsed | | | 0.384 | 1.000 | 0.555 | [[0, 143], [0, 89]] |

The **random forest** is selected (best CV F1) and meets the requirement of lapsed-class test F1 >= 0.55 (0.637; asserted in code). Honest reading: the gain over "flag everyone" (F1 0.555) is modest, because the synthetic features carry limited signal. Business trade-off: contacting customers costs money, so precision matters for expensive offers, while recall matters when a lost customer is costly; here both sit near 0.64.

**Class-balance experiment** (oversample the **training** set only: all 924 originals kept, minority rows duplicated with replacement, `random_state=42`, up to 570 vs 570; tuned parameters refitted once, test set unchanged):

| Model | Precision | Recall | F1 | Confusion matrix |
|---|---|---|---|---|
| Logistic, oversampled | 0.589 | 0.854 | 0.697 | [[90, 53], [13, 76]] |
| Random forest, oversampled | 0.558 | 0.809 | 0.661 | [[86, 57], [17, 72]] |

Oversampling trades precision for recall. It was *not* used to pick the selected model (the selection rule was fixed beforehand); if the business wants high recall it is a legitimate next step.

**Persistence and retention list.** The selected pipeline is saved to `outputs/lapse_model.joblib`; reloading gives identical test predictions (asserted). `outputs/retention_list.csv` lists the 10 Premium customers in the test set with the highest predicted lapse probability (`customer_id, lapse_probability, actually_lapsed`). **Caveat:** the Premium flag comes from RFM over the *whole* April-September window, so it is a post-hoc label that already reflects September behaviour. The list is **illustrative, not a leakage-free evaluation**. Only 1 of its 10 customers actually lapsed, as expected for the most loyal group.

## Part C — Support Ticket Triage

164 tickets (156 with usable text, 4 empty, 4 over-long). `mock_llm.chat` is used for classification; categories are billing, delivery, technical, fraud.

| Check | Expected | Observed |
|---|---|---|
| Rows in `outputs/triage_results.csv` | 164 | 164 |
| `classified_json` / `classified_keyword` / `flagged_invalid_output` | 135 / 12 / 9 | 135 / 12 / 9 |
| Simulated transient failures recovered by retry | 19 | 19 (0 unrecovered) |
| Empty / over-long tickets blocked without a model call | 4 / 4 | 4 / 4 (0 tokens, 0 attempts) |

**Zero-shot vs few-shot** (accuracy against `true_category`, counting the 156 tickets sent to the model): few-shot **94.2%** (147/156; 100% of the 147 it classified), zero-shot **80.8%** (126/156; 85.7% of classified). Few-shot costs 12,520 tokens versus 7,372. The 9 invalid outputs (`"shipping"`) are identical in both and go to a human. I have no brief reference value for these accuracies.

- **PII protection:** email, then phone, then order id are masked *before* any model call (55 emails, 91 phones, 132 order ids masked). `assert_no_pii` checks **every** message, system and user, before every call including retries; tests capture all 175 prompts and confirm no raw email, phone or order id appears.
- **Empty and long tickets:** blank text and tickets over 400 estimated tokens are never sent to the model; they go to human review.
- **Retries:** up to 3 retries with exponential backoff (0.05 s x 2^n) and +-jitter; counters for calls (175), retries (19), failures, tokens and backoff seconds are in `outputs/triage_summary.json`.
- **Budget enforcement:** before each call the projected tokens and cost must fit `token_budget` (100,000) and `cost_budget_usd` (0.05); otherwise the ticket is marked `budget_exhausted` and handed to a human. Demo with a 1,000-token cap: 11 tickets classified, 145 `budget_exhausted`.
- **Model cost:** 12,520 tokens, about **USD 0.0063**. The price (USD 0.50 per 1M tokens) is an illustrative assumption because the mock is free; it is a constant in `src/triage.py`.
- **Routing rules** (Part B segments joined by `customer_id`): guard/invalid/budget outcome -> `human_review`; fraud -> `security_team` (HIGH); Premium customer -> `premium_desk` (HIGH); else `support_queue`. Observed: support_queue 102, security_team 34, human_review 17, premium_desk 11. `validate_results` re-derives every route independently. If `customer_segments.csv` is missing, the pipeline fails loudly instead of silently routing without segments.
- **Human handoff:** 17 tickets (4 empty, 4 long, 9 invalid output), each with a `handoff_reason` column.

## Part D — Policy RAG Assistant and Refund Agent

**Ingestion and retrieval.** Five policy documents, 40-word chunks, 10-word overlap -> refund 4, delivery 3, escalation 3, payment 3, privacy 2 = **15 chunks** (matches the reference). Retrieval is scikit-learn `TfidfVectorizer` + cosine similarity. `answer()` sends exactly two messages: a system message beginning `POLICY ASSISTANT` and a user message with `CONTEXT:` and `QUESTION:`.

| Evaluation | Result |
|---|---|
| Expected document retrieved (top-3 / top-1), 12 supported questions | 12/12 / 12/12 |
| Expected fact in the answer | 12/12 |
| Unsupported questions give the fallback "I could not find that information in the policy documents." | 2/2 |
| False fallbacks on supported questions | 0 |

**Zero-overlap comparison** (same 40-word chunks, overlap 0 -> 13 chunks): the right *document* is still retrieved 12/12, but only **10/12 facts** are found. Two answers fail because the fact is cut across a chunk boundary: *"What delivery time makes a completed order eligible for a full refund?"* (refund policy) and *"How soon is money returned when a payment fails but is deducted?"* (payment policy). So the failure observed is an *answer* failure caused by a split fact, not a document-ranking failure. See `outputs/rag_zero_overlap_failures.csv`.

**Tools** (`TOOL_MAP`): `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, `escalate_to_human`. Policy: a completed order delivered in **more than 45 minutes** gets a full refund; up to INR 1,000 is automatic; above needs a human. Safety:
- `issue_refund` and `escalate_to_human` **re-check the order in SQLite themselves** (exists, completed, late, amount equals the amount due, within or above the INR 1,000 limit) and raise otherwise, whatever the model asked for; every rejection is written to the ledger.
- Both are **idempotent per order**: the ledger is consulted first, so a repeated call cannot pay twice.
- Every executed side effect is appended and fsynced to `outputs/refund_ledger.jsonl` (audit trail).

**ReAct loop.** System marker `AGENT`; at most 6 iterations (the escalation policy's 6 steps), a token limit (default 6,000) and a time limit (default 10 s). Unknown tools or bad arguments end in `FALLBACK_TOOL_ERROR`; an unauthorised refund ends in `BLOCKED_UNAUTHORIZED`. After each tool the run writes an atomic checkpoint (`state/<run_id>.json`: messages, tokens, executed tools, steps, elapsed time); `resume_agent(run_id)` continues from it, and resuming a finished run is a no-op.

**Scenarios** (orders are derived from the cleaned data: the lowest `order_id` matching each description; the expected amounts identify them):

| Scenario | Expected | Observed | Model calls | Tools |
|---|---|---|---|---|
| S1 (ORD-100048) | REFUND INR 128.25 | REFUND 128.25 | 5 | get_order, search_policy, calculate_refund, issue_refund |
| S2 (ORD-100067) | REFUND INR 135.00 | REFUND 135.00 | 5 | same |
| S3 (ORD-100145) | ESCALATE INR 1999.00 | ESCALATE 1999.00 | 5 | ... escalate_to_human |
| S4 cancelled order | DENIED | DENIED | 3 | get_order, search_policy |
| S5 delivered on time | DENIED | DENIED | 4 | ... calculate_refund (0.00) |
| S6 unknown order | NOT_FOUND | NOT_FOUND | 2 | get_order |
| S7 `[LOOP]` | FALLBACK_MAX_ITERATIONS | FALLBACK_MAX_ITERATIONS | 6 | search_policy x 6 |

**Demonstrations** (private ledger, `outputs/agent_demos.json`):
- *Token limit 200:* `FALLBACK_TOKEN_LIMIT` after 354 tokens, before any refund; none issued.
- *Timeout 0:* `FALLBACK_TIMEOUT` with 0 model calls.
- *Crash after the 2nd tool call, then resume:* the checkpoint holds `get_order`, `search_policy`; no money moved. Resume executes `calculate_refund`, `issue_refund` only, so the full sequence is exactly `get_order, search_policy, calculate_refund, issue_refund`; result REFUND **INR 128.25**, 5 model calls in total, **exactly one** `issue_refund` ledger entry.
- *Harder case, crash after payment but before the checkpoint:* the resumed run repeats `issue_refund`, the ledger blocks it (`duplicate_prevented`), and there is still one payment.

<!-- react-start -->
**ReAct versus plan-execute.** In ReAct the model alternates a thought, one tool call and an observation, deciding the next step only after seeing the last result. That suits the refund task: the right branch depends on data that is unknown up front (order not found, cancelled, on time, above the INR 1,000 limit), so each observation can redirect the run. It also gives a natural audit trail and clean checkpoints, because every step is one tool call. The costs are more model calls (S1 needs five), a context that grows with every observation, and a risk of looping, which is why the loop is capped at 6 iterations with token and time limits. Plan-execute writes the whole plan first and then runs it with little or no further model involvement. It is cheaper and more predictable for fixed pipelines, and it is easier to review a plan before anything executes. Its weakness is brittleness: if step two reveals the order is cancelled, a fixed plan has to be abandoned or re-planned. For a small, branching, safety-critical workflow, ReAct with hard limits and tool-level authorization is the better fit; for long, well-understood jobs, plan-execute wins on cost.
<!-- react-end -->

## Web App — Customer Portal and Admin Console

A Flask front end on top of the Part A–D work (`webapp/`). It runs offline on `http://127.0.0.1:5000`.

```bash
pip install -r requirements.txt            # adds flask and openpyxl
python run_project.py                      # Parts A-D first: the web app reads data/instamart.db and outputs/customer_segments.csv
python -m webapp.manage init               # once: users, credential files, Excel record, SQL dumps
python -m webapp.app                       # then open http://127.0.0.1:5000
python -m webapp.manage restore            # fresh clone: rebuild data/app.db from data/dump/app_dump.sql
python -m webapp.manage reset-password USERNAME   # new random password, printed once
```

| Role | Can do |
|---|---|
| **Customer** (`user01`-`user10` demo logins, or self-register at `/register`) | Order products, cancel an order while it is still "placed", see own order history, raise support tickets (optionally linked to one of their own orders), follow up on their tickets |
| **Admin** (`admin`; cannot self-register) | See every customer (search by ID/name/username, filter by city and segment) with profile, segment, order history, web orders and tickets; update order status; reply to, answer and close tickets; download the Excel record |

**Data and records**
- Web data lives in a separate `data/app.db` (users, web customers, orders, tickets, messages), so the 12 Part A checkpoints are unaffected. The Part A database is attached **read-only**.
- **Excel record:** `data/records/instamart_web_records.xlsx` (sheets Users, WebCustomers, Orders, Tickets, TicketMessages) is rewritten after every change. It has **no password column**, and text cells beginning with `=`, `+`, `-` or `@` are neutralised against spreadsheet formula injection.
- **SQL dumps** (committed, for sharing via Git): `data/dump/app_dump.sql` (web data, including password *hashes*) and `data/dump/instamart_dump.sql` (the cleaned Part A database). `data/app.db` itself is git-ignored; restore it from the dump.
- **Credentials** are written to two separate git-ignored files, `credentials/ADMIN_CREDENTIALS.txt` and `credentials/USER_CREDENTIALS.txt`. Passwords are random and stored only as PBKDF2-SHA256 hashes; they never appear in the Excel file, dump or code. Share the two files through different channels, and rotate them with `reset-password` before any real use.

**Ticket handling:** a new ticket is classified by the Part C pipeline (email, phone and order IDs masked **before** the mock model sees the text), then routed with the Part C rules (fraud to `security_team`, Premium to `premium_desk`, doubtful output to `human_review`). The customer's own words are stored as written, so admins see the original text; only the model input is masked.

**Security measures** (each covered by `tests/test_webapp.py`): hashed passwords; signed HttpOnly SameSite cookies; per-session CSRF token on every POST; login lock-out after 5 failures in 10 minutes; admin routes return 403 to customers; a customer asking for another customer's ticket or order gets 404; prices come from the database, never the form; parameterised SQL; auto-escaped templates plus a strict Content-Security-Policy; no external scripts or fonts.


**Sharing the app over the internet with ngrok (run on YOUR machine; not possible from the build sandbox).**

Shortest route: after `ngrok config add-authtoken <YOUR_TOKEN>`, run `python -m webapp.share`. It starts the app and the tunnel together and prints the public login URL. The manual steps are below.
```bash
# one-time: install ngrok and add your own token from https://dashboard.ngrok.com
ngrok config add-authtoken <YOUR_NGROK_TOKEN>

# terminal 1 (Windows PowerShell: $env:INSTAMART_PUBLIC_HTTPS="1"; python -m webapp.app)
INSTAMART_PUBLIC_HTTPS=1 python -m webapp.app

# terminal 2: ngrok prints a https://<random>.ngrok-free.app URL; share <that URL>/login
ngrok http 5000
```
`INSTAMART_PUBLIC_HTTPS=1` makes the session cookie HTTPS-only and trusts ngrok's forwarded headers. Before exposing the app: run `python -m webapp.manage reset-password` for `admin` and every demo user (the shipped passwords came from the build run), keep the tunnel open only while testing, and consider adding `--basic-auth "name:long-password"` to the `ngrok http` command as a second gate. ngrok's free plan may show a one-time warning page before the login page. Anyone with the URL can reach the login form, and the admin account can read all customer data, so do not leave it running unattended.


### Keeping the app running

`python -m webapp.app` stops when its terminal is closed or interrupted, and then the page does not load. `python run_forever.py` (on Windows, double-click `start_app.bat`) does the first-time setup if needed, opens the login page, and starts the app again whenever it stops; it gives up with a message if the app dies five times in a row right after starting (usually another program using port 5000, which `INSTAMART_PORT` can change). Logins survive restarts because users live in `data/app.db` and the session key in `data/.secret_key` (tested in `tests/test_support_bot.py`). It only runs while the computer is on and that window stays open. To start it at every Windows sign-in, run once: `schtasks /Create /SC ONLOGON /TN InstamartCopilot /TR "\"<full path>\start_app.bat\""`. For a page that stays up without your computer, use the deployment below.

### Help chat and support bot (bot first, ticket only if it cannot resolve)

Customers do not fill in a ticket form. A floating **💬 Help** bubble opens a chat with the support bot (`webapp/support_bot.py`, `webapp/static/chat.js`), which runs fully offline and tries to fix the problem itself, in conversation, before any person is involved:

- **Order problems** (late or missing delivery, refund, cancel, wrong or expired items, rude rider, double charge): it looks up the customer's own orders (web and history), assumes the order when only one is live, otherwise lists the recent ones and asks which one (the customer replies with `1`, `2`, an order id, or "latest"). It then works from the order's real status and age against the policy: inside the 30-minute promise it tells the customer to wait and keep the phone nearby; between 30 and 60 minutes it explains the heavy-rain allowance and how long to wait; a placed order can be cancelled from My orders; a cancelled order was never charged; a completed order delivered in more than 45 minutes is entitled to a full refund (with the ₹1000 approval rule and the 7-day and 5-working-day limits).
- **Policy questions** ("What is the delivery fee on small orders?") get the matching policy sentences quoted verbatim with their section (sentence-level TF-IDF with cosine similarity). A vague message first gets one clarifying question.
- **Payment failed** gets the 48-hour automatic-return rule.

It reads only masked text for retrieval, and every order fact comes from the database rows, not from the message. After an answer the customer can press **Solved, thank you** (logged in `chat_log`) or **Not solved – create a ticket**.

A person is needed only when the bot cannot fix it itself: it cannot issue refunds, speed up a rider, approve compensation or reverse a double charge; or the customer says it is still not solved ("still not received"); asks for a person; the topic is fraud; the answer is not in the policies after a clarifying question; or it has replied `MAX_BOT_REPLIES` (6, from the escalation policy) times. Each hand-over tells the customer what the agent will be told (order, status, amount, what they said). The ticket carries the whole conversation, its subject is the customer's latest message, and it appears in the admin queue as `open`. `POST /tickets` no longer exists; tickets are created only by `POST /chat/ticket`. `tests/test_support_bot.py` covers each path with fixed orders and a fixed clock.

Limits: the bot can only repeat what the policy files say, and it does not look up refunds or move money. The dialogue rules are keyword patterns written by hand, not a trained model, so unusual wording can fall through to the clarifying question or a hand-over. `MIN_SCORE` was chosen by hand from the 12 answerable questions in `data/raw/policy_eval.csv` (they score 0.36 or more) and a few vague complaints (0.25 or less); it was not tuned on real tickets, and 1 of those 12 questions still retrieves the wrong sentence. Databases created earlier are upgraded in place. Bot messages are authored by a disabled `copilot_bot` account that cannot log in.

### Public deployment (worldwide URL)

A URL that works from any computer needs the app hosted on a public server, which the development sandbox cannot do. The repository includes what a host needs: `webapp/wsgi.py` (gunicorn entry point that rebuilds `data/app.db` from `data/dump/app_dump.sql` on first start), `Procfile`, and `render.yaml`. On Render: *New + → Blueprint →* choose this repository; the service name `instamartcopilot` gives `https://instamartcopilot.onrender.com` when that name is free, and it sets `INSTAMART_PUBLIC_HTTPS=1` and a generated `INSTAMART_SECRET_KEY`. The free tier has no persistent disk, so tickets and orders created online are lost when the service restarts; the logins in the committed dump (the same ones as in your `credentials/` files) come back each time. A custom domain such as `instamartcopilot.com` has to be bought and attached in the host's settings. Alternatively `python -m webapp.share` publishes your local copy through ngrok while your computer is on.\n\n**Hosted demo (claude.ai page).** `hosted_demo/index.html` is a self-contained re-implementation of the portal and admin console for sharing a live link without running anything locally. It is **not** the Flask app: logins are checked in the browser against PBKDF2-SHA256 hashes in the page's shared database, those hashes are readable by anyone who can open the page, and the keyword triage is a stand-in for the Part C model, and its help chat is a port of the Flask one that reads the same policy text. It lives at a claude.ai address that cannot be renamed. Use it with fictional data only. Viewers need Contributor access to place orders or reply to tickets. The accounts are seeded with the owner's own demo passwords, which are separate from the Flask app's credentials.

**Web app limitations:** development server only (use a production WSGI server and HTTPS for real use; set `INSTAMART_SECRET_KEY`); orders have no payment, stock or address handling; web orders are kept in `data/app.db` and are not merged into the Part A `orders` table or the ML features; login lock-out is per username, not per IP; one admin account; no password-change or email-reset screen yet.

## Limitations and Next Steps

- **Brief not available.** Raw-audit figures, cleaned row counts, RFM statistics, the exact K-means/GridSearch specifications (parameter grids, scoring) and the expected SQL results from the original brief could not be checked, nor the zero-/few-shot accuracy references. Grids and F1 scoring follow the pre-existing code. Compare the observed values above with the brief and report any difference.
- **Scenario orders are a choice.** S1-S3 are identified by their expected amounts and the lowest matching `order_id`; the brief may name specific orders.
- **Retention list is illustrative** (post-hoc Premium flag, test-set rows only), and `age` is imputed with city medians computed over all customers (no target information, but not strictly train-only).
- **Synthetic data and mock model.** Lapse features carry limited signal (F1 0.637 vs 0.555 for "always lapsed"). The mock LLM is rule-based, so accuracies say little about a real model; the cost price is an assumption.
- **Local simulation only.** The ledger is a JSONL file, not a payment system. Checkpoint resume restarts the wall-clock timer but carries the elapsed time forward.
- **Notebook validation.** Jupyter/nbclient was not installed, so the notebook was verified with `tests/run_notebook.py` (a clean interpreter running every code cell in order), not a real kernel. Open it in VS Code/Jupyter and *Restart & Run All* to confirm in your environment.
- **Line endings.** `generate_data.py` writes CSVs with the platform's line endings, so `git status` may show `data/raw/*.csv` as modified after regeneration although the content is identical.
- **Stale copies.** `notebooks/data/` holds older, different copies of the raw data and policies that nothing now uses; they are left in place for you to decide on.
- **Next steps:** tune on recall (or use cost-sensitive learning) for retention campaigns; evaluate segments with a held-out time window; replace the mock with a real model behind the same interface; add a human-approval queue for escalations; add data contracts and drift monitoring.
