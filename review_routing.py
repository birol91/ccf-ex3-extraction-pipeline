"""
Exercise 3 — Step 5: Field-level confidence + human-review routing.

Two new ideas, both exam favourites.

1. FIELD-LEVEL confidence, not document-level.
   The model scores each field 0..1. We route a document to a human only when a
   specific field is below threshold, and we tell the human WHICH fields are
   weak so they only check those — not the whole document. A document-level
   score would hide the one shaky field among nine solid ones.

2. FIELD-LEVEL accuracy analysis, not aggregate.
   The trap: "95% overall accuracy" sounds great and can be a lie. That 95%
   can hide "invoice / total_amount is only 40% correct" — a number you'd ship
   to accounting. So analyze_accuracy() breaks accuracy down by
   (doc_type x field), never a single aggregate. The whole lesson of this step
   is that the aggregate MASKS the weak cells; you must look at the breakdown.
"""

import os

from anthropic import Anthropic
from dotenv import load_dotenv

from schema import (
    ExtractedInvoiceWithConfidence,
    extraction_tool_with_confidence,
)

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

client = Anthropic()
MODEL = "claude-sonnet-4-6"

# Below this, a field is "not trustworthy enough to auto-accept". Tune against a
# labelled set: too high floods the human queue, too low lets errors through.
CONFIDENCE_THRESHOLD = 0.7


def extract_with_confidence(document_text: str) -> ExtractedInvoiceWithConfidence | None:
    """Run extraction with the confidence-aware tool/schema (one-shot)."""
    resp = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        tools=[extraction_tool_with_confidence],
        tool_choice={"type": "tool", "name": "extract_invoice_with_confidence"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract the invoice data with an honest per-field confidence "
                    f"score. Be conservative.\n\n{document_text}"
                ),
            }
        ],
    )
    tool_block = next(b for b in resp.content if b.type == "tool_use")
    try:
        return ExtractedInvoiceWithConfidence(**tool_block.input)
    except Exception:
        return None


def route_extraction(
    extracted: ExtractedInvoiceWithConfidence,
) -> tuple[str, list[str]]:
    """
    Decide auto-accept vs human-review based on per-field confidence.

    Returns ("auto_accept", []) when every field clears the threshold, or
    ("human_review", [low_field, ...]) listing only the fields that need a
    human. Surfacing just the weak fields keeps the reviewer's job small.
    """
    low_conf_fields = [
        name
        for name, field in extracted.model_fields.items()  # iterate field names, then getattr
        if getattr(extracted, name).confidence < CONFIDENCE_THRESHOLD
    ]
    if low_conf_fields:
        return "human_review", low_conf_fields
    return "auto_accept", []


def analyze_accuracy(results: list[dict], ground_truth: list[dict]) -> dict:
    """
    Compute accuracy broken down by (doc_type x field) — NOT a single number.

    Inputs are aligned lists of dicts. Each dict is a flat field->value mapping;
    each results dict additionally carries "doc_type" so we can bucket by it.

        results[i]      = {"doc_type": "invoice", "invoice_number": "X", ...}
        ground_truth[i] = {"invoice_number": "X", ...}

    Returns:
        {doc_type: {field: {"correct": int, "total": int, "accuracy": float}}}

    Why this shape: it exposes exactly the cell the aggregate would hide. Scan
    the returned table for low-accuracy (doc_type, field) pairs — those are
    where the pipeline is actually weak, regardless of how good the headline
    number looks.
    """
    breakdown: dict = {}

    for pred, truth in zip(results, ground_truth):
        doc_type = pred.get("doc_type", "unknown")
        per_field = breakdown.setdefault(doc_type, {})

        # Compare every ground-truth field for this document.
        for field, true_val in truth.items():
            cell = per_field.setdefault(field, {"correct": 0, "total": 0})
            cell["total"] += 1
            # String-compare normalized values; extend with field-specific
            # tolerance (e.g. numeric rounding) as needed.
            if _norm(pred.get(field)) == _norm(true_val):
                cell["correct"] += 1

    # Finalize accuracy per cell.
    for fields in breakdown.values():
        for cell in fields.values():
            cell["accuracy"] = cell["correct"] / cell["total"] if cell["total"] else 0.0

    return breakdown


def _norm(v) -> str:
    """Normalize a value for comparison: None -> '', strip + lowercase strings."""
    if v is None:
        return ""
    return str(v).strip().lower()


if __name__ == "__main__":
    # Demonstrate the masking effect: aggregate would read ~83%, but the
    # breakdown reveals invoice/total_amount is the weak cell.
    results = [
        {"doc_type": "invoice", "invoice_number": "A1", "total_amount": "100"},
        {"doc_type": "invoice", "invoice_number": "A2", "total_amount": "999"},  # wrong
        {"doc_type": "receipt", "invoice_number": "R1", "total_amount": "5"},
    ]
    truth = [
        {"invoice_number": "A1", "total_amount": "100"},
        {"invoice_number": "A2", "total_amount": "250"},
        {"invoice_number": "R1", "total_amount": "5"},
    ]
    import json

    print(json.dumps(analyze_accuracy(results, truth), indent=2))
