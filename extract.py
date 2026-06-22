"""
Exercise 3 — Step 2: Validation-retry loop with error classification.

This is the reliability core of the pipeline, and the single most exam-relevant
idea in the whole exercise. The model WILL sometimes produce output that fails
Pydantic validation. What we do next depends on WHY it failed:

  RESOLVABLE error  — a FORMAT problem. The information is in the document, the
                      model just shaped it wrong: a number came back as a
                      string, an enum value is misspelled, a field has the
                      wrong type. Sending the document + the bad output + the
                      specific error back to the model CAN fix this. -> retry.

  UNRESOLVABLE error — the information is genuinely ABSENT from the source. A
                      required field (invoice_number) is missing because the
                      document truly does not contain one. No amount of retrying
                      conjures data that isn't there; retrying just burns money
                      and risks the model fabricating to satisfy us. -> stop,
                      route to human review.

classify_error() is the dividing line. The trap the exam sets is the infinite
retry loop: code that retries on EVERY ValidationError eventually either loops
to the retry cap on unresolvable errors (wasting calls) or, worse, pressures
the model into inventing a value. We avoid both by classifying first.

The follow-up message on a resolvable retry carries three things, all required
to make the model actually correct itself:
    1. the original document (so it can re-read the source),
    2. its own bad output (so it sees what it produced),
    3. the SPECIFIC validation error (so it knows exactly what to fix),
plus a reminder: do not fabricate absent information.

Model: claude-sonnet-4-6 (extraction is high-volume, latency-tolerant — Sonnet
is the cost/intelligence sweet spot for this workload, per the brief).
"""

import os

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import ValidationError

from schema import ExtractedInvoice, extraction_tool

# Load ANTHROPIC_API_KEY from the project-root .env (two levels up), matching Ex1/Ex2.
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment

MODEL = "claude-sonnet-4-6"

# The required fields of the schema. If validation fails *because one of these
# is missing*, that is the unresolvable case (info absent from source).
_REQUIRED_FIELDS = {
    name
    for name, field in ExtractedInvoice.model_fields.items()
    if field.is_required()
}


def classify_error(validation_error: ValidationError) -> str:
    """
    Decide whether a Pydantic ValidationError is worth retrying.

    Returns "resolvable" or "unresolvable".

    Logic:
      - A 'missing' error on a REQUIRED field means the model could not find the
        field in the source. That is information genuinely absent -> unresolvable.
      - Anything else (wrong type, bad enum value, value out of range, malformed
        number) is a formatting mistake the model can correct on a second look
        -> resolvable.

    If even one error in the batch is unresolvable, we treat the whole
    extraction as unresolvable: a missing required field can't be papered over
    by fixing the format of other fields.
    """
    for err in validation_error.errors():
        # err["loc"] is a tuple path, e.g. ("invoice_number",) or
        # ("line_items", 0, "amount"). The top-level field name is loc[0].
        top_field = err["loc"][0] if err["loc"] else None
        is_missing = err["type"] in ("missing", "value_error.missing")
        if is_missing and top_field in _REQUIRED_FIELDS:
            return "unresolvable"
    return "resolvable"


def _route_to_human_review(document_text: str, reason: str, last_output=None) -> None:
    """
    Stand-in for the human-review queue. In production this would enqueue the
    document (and any partial extraction) for a person. Here we just log it so
    the control flow is visible. Returning None from extract_with_retry signals
    "this one needs a human" to the caller.
    """
    preview = document_text[:80].replace("\n", " ")
    print(f"[HUMAN REVIEW] reason={reason} doc={preview!r} partial={last_output}")


def extract_with_retry(
    document_text: str, max_retries: int = 2
) -> ExtractedInvoice | None:
    """
    Extract an invoice, retrying only on resolvable (format) errors.

    Returns a validated ExtractedInvoice on success, or None when the document
    is routed to human review (unresolvable error, or retries exhausted).
    """
    messages = [
        {
            "role": "user",
            "content": f"Extract the invoice data from this document:\n\n{document_text}",
        }
    ]

    for attempt in range(max_retries + 1):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            tools=[extraction_tool],
            # Force the tool call so we always get schema-shaped output.
            tool_choice={"type": "tool", "name": "extract_invoice"},
            messages=messages,
        )

        tool_block = next(b for b in resp.content if b.type == "tool_use")

        try:
            return ExtractedInvoice(**tool_block.input)  # validated — done.
        except ValidationError as exc:
            kind = classify_error(exc)

            if kind == "unresolvable":
                # Information is absent from the source. Retrying cannot help and
                # risks fabrication. Stop immediately, hand to a human.
                _route_to_human_review(
                    document_text,
                    reason="unresolvable (required info absent from source)",
                    last_output=tool_block.input,
                )
                return None

            # Resolvable (format) error. If we're out of retries, give up to a
            # human rather than loop forever.
            if attempt == max_retries:
                _route_to_human_review(
                    document_text,
                    reason="resolvable but retries exhausted",
                    last_output=tool_block.input,
                )
                return None

            # Build the corrective follow-up: document context is already in
            # `messages`; we add the model's bad output, then a tool_result
            # carrying the SPECIFIC error and the no-fabricate reminder.
            messages.append({"role": "assistant", "content": resp.content})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": (
                                f"Your extraction failed validation:\n{exc}\n\n"
                                "Fix ONLY the fields named in the error, re-reading "
                                "the document above. Do not change valid fields. "
                                "If a field's information is not in the document, "
                                "leave it null — do NOT fabricate."
                            ),
                            "is_error": True,
                        }
                    ],
                }
            )

    return None  # defensive; loop always returns inside the body.


if __name__ == "__main__":
    sample = (
        "INVOICE #INV-2024-0042\n"
        "Vendor: Acme Corp\n"
        "Bill To: Globex LLC\n"
        "Date: 2024-03-01  Due: 2024-03-31\n"
        "Total: $1,250.00 USD\n"
    )
    result = extract_with_retry(sample)
    print(result.model_dump_json(indent=2) if result else "-> routed to human review")
