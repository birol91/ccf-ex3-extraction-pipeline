# Structured Data Extraction Pipeline

An end-to-end invoice extraction pipeline built with `claude-sonnet-4-6` and Pydantic. The goal is a **reliable system, not a perfect model** — the model will make mistakes; the pipeline catches them, retries where possible, and routes the rest to a human reviewer.

> A learning project (CCA-F Preparation Exercise 3). Covers Domain 4 (Prompt Engineering & Structured Output) and Domain 5 (Context Management & Reliability). Target exam questions: Q11, Q12.

---

## The Pipeline

See [`architecture.html`](architecture.html) for the full visual diagram.

```
PDF document
    │
    ▼
ingest.py  ──  opendataloader-pdf (Java 11+) → plain text
    │
    ▼
extract.py  ──  tool_use + JSON schema (schema.py) → structured output
    │
    ├── Pydantic validation passes?
    │       ├── YES → confidence routing (review_routing.py)
    │       │             ├── all fields high-confidence → auto-accept
    │       │             └── any field low-confidence  → human review queue
    │       │
    │       └── NO (validation error)
    │               ├── RESOLVABLE (format mismatch) → retry with error context
    │               └── UNRESOLVABLE (info absent from source) → human review
    │
    ▼  (scale)
batch_runner.py  ──  Message Batches API, 100 docs, custom_id tracking
```

---

## Three Engineering Decisions

These are the design choices the exam tests directly. Read them before the code.

### 1. Nullable fields + "never fabricate" instruction

Fields that may be absent in a real document (`due_date`, `vendor`, `tax_id`) are declared `Optional[...] = None` in the schema. The extraction tool description explicitly states: **"If a field's information is not present in the source, return null — never fabricate a value."**

Why it matters: if you mark an absent field as `required`, the model fills the gap with a plausible-sounding but invented value. The pipeline then stores a fabrication as fact. A nullable field + explicit null instruction prevents this. The `sample_docs/invoice_missing_fields.txt` test document exists specifically to verify this behaviour — run it and confirm every absent field returns `null`, not a guess.

### 2. Resolvable vs. unresolvable retry — avoiding the infinite loop

Not all validation failures are equal:

| Error type | Example | Retry fixes it? |
|---|---|---|
| **Resolvable** (format mismatch) | `total_amount` returned as `"$8,400.50"` (string) instead of `8400.50` (float) | Yes — send the error back with specific correction instructions |
| **Unresolvable** (information absent) | `vendor` is genuinely missing from the document | No — the model cannot conjure information that does not exist |

Retrying an unresolvable error is an infinite loop that burns tokens and never converges. The retry loop in `extract.py` classifies each `ValidationError` before deciding whether to retry or escalate to human review. This classification is the most important logic in the pipeline.

### 3. Field-level accuracy, not aggregate

A 95% aggregate accuracy score sounds healthy. It hides the fact that one combination — say, `invoice / total_amount` — might be correct only 40% of the time. `review_routing.py` builds an accuracy table keyed by `(doc_type, field_name)`. Each cell is measured independently. Weak cells are visible and actionable. An aggregate number is not.

---

## Batch vs. Sync Decision Tree

```
New extraction request arrives
         │
    Is someone waiting for the result?
    (user-facing, CI pipeline, blocking flow)
         ├── YES → Synchronous API (respond immediately)
         └── NO  → Is it latency-tolerant? (nightly job, bulk upload)
                        └── YES → Message Batches API
                                  - ~50 % cost reduction
                                  - Results within 24 h
                                  - Track each doc with custom_id
                                  - NOTE: batch does NOT support multi-turn
                                    tool calling. Run validation-retry
                                    outside the batch, on failed custom_ids.
```

The `batch_runner.py` module handles the latency-tolerant path. When results arrive, only failed `custom_id`s are resubmitted — never the whole batch. Oversized documents are chunked before resubmission.

---

## Repo Layout

```
ex3-extraction-pipeline/
├── README.md
├── requirements.txt
├── architecture.html          # visual pipeline diagram
├── schema.py                  # ExtractedDocument + FieldWithConfidence models
├── ingest.py                  # opendataloader-pdf wrapper → plain text
├── extract.py                 # tool definition, validation-retry loop
├── fewshot.py                 # varied-format few-shot examples
├── batch_runner.py            # Message Batches submit + custom_id result handling
├── review_routing.py          # confidence routing + field-level accuracy analysis
└── sample_docs/
    ├── invoice_narrative.txt       # amounts and fields embedded in prose
    ├── invoice_table.txt           # same invoice, tabular line-item format
    ├── invoice_missing_fields.txt  # vendor and due_date intentionally absent
    └── invoice_other_type.txt      # credit note — triggers doc_type = "other"
```

---

## Run It

**Prerequisites**

- Python 3.10+
- Java 11+ (required by opendataloader-pdf for real PDF ingestion; `.txt` test docs bypass this)
- An Anthropic API key

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Set your API key (never commit this)
cp .env.example .env
# edit .env and add: ANTHROPIC_API_KEY=sk-ant-...

# 3. Extract a single document (sync)
python extract.py sample_docs/invoice_narrative.txt

# 4. Run the full batch (100 docs)
python batch_runner.py --input docs/ --batch-id-out batch_id.txt

# 5. Collect batch results and route for review
python review_routing.py --batch-id-file batch_id.txt
```

> **Java note:** `ingest.py` shells out to opendataloader-pdf. If Java is not on your `PATH`, ingestion falls back to reading `.txt` files directly. All four `sample_docs/` files are plain text and work without Java.

### Sample output (live run, `claude-sonnet-4-6`)

Extraction over `sample_docs/`, showing the two key behaviours:

```
invoice_narrative.txt   →  "eight thousand four hundred dollars and fifty cents" parsed to
                           total_amount = 8400.5; vendor, date, line items all extracted.
invoice_missing_fields  →  vendor and due_date are ABSENT in the source →
                           returned as null (not fabricated).        ← fabrication prevention
invoice_other_type.txt  →  doc_type = credit_note (correct enum), negative line amount -1200.0.
```

Every result is a valid Pydantic object — `tool_use` guarantees the structure; the schema's
"if absent, null — never fabricate" instruction keeps the model from inventing missing values.

---

## Anti-Patterns

| Anti-pattern | What breaks |
|---|---|
| Mark absent fields as `required` | Model fabricates plausible but false values to satisfy the schema |
| Retry on unresolvable errors (info absent) | Infinite loop; tokens wasted; result never improves |
| Report only aggregate accuracy | A 95% headline hides a 40% failure rate on a specific field |
| Parse JSON from free text instead of using `tool_use` | Model can produce malformed JSON; `tool_use` enforces the schema |
| Omit `"other"` from the `doc_type` enum | Model force-fits unknown document types into a wrong category |
| Route a blocking request through the Batch API | Batch has no latency guarantee; caller waits up to 24 h |
| Expect multi-turn retry inside a batch request | Batch is one-shot per request; retry loop must run outside |
| Resubmit the whole batch on partial failure | Wastes cost; only failed `custom_id`s need resubmission |
| Auto-accept low-confidence extractions | Propagates uncertain data downstream without a human check |

---

## Security Note

- Real secrets never go in committed files. Use `.env` (listed in `.gitignore`).
- `.env.example` contains key names only — no values.
- The `ANTHROPIC_API_KEY` is read at runtime via `python-dotenv`; it is never hard-coded.

---

See [`architecture.html`](architecture.html) for the full visual diagram of the pipeline stages.
