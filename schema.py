"""
Exercise 3 — Step 1: Extraction schema (INVOICE-focused).

This file defines the Pydantic models that shape every extraction in the
pipeline, plus the tool definition we force Claude to call. The schema is the
single most important design surface in the whole exercise: a good schema is
what makes the model return null instead of fabricating, pick "other" instead
of mislabelling, and produce output we can validate for free.

Three patterns the exam tests directly live here:

  1. Required vs optional/nullable fields.
     - `invoice_number` is REQUIRED — every invoice has one.
     - vendor, customer, totals, dates, line items are Optional → null when
       absent. Each carries an "if absent return null, never fabricate"
       description. The description is not decoration: the model reads it, and
       repeating "never fabricate" both in the field docs AND the tool
       description measurably suppresses made-up values.

  2. Enum + "other" + detail escape valve.
     - `doc_type` is a closed Literal set, but includes "other" so the model
       never has to jam an unexpected document into a wrong category. When it
       picks "other" it explains itself in `doc_type_detail`.

  3. Field-level confidence (Step 5).
     - `ExtractedInvoiceWithConfidence` wraps every field in a value+confidence
       pair so the routing layer can flag low-confidence fields for a human.

Why tool_use and not "ask for JSON": forcing a tool call guarantees the output
conforms to this schema's JSON Schema. Free-text JSON can come back with a
trailing comma or an unescaped quote and blow up json.loads(). tool_use removes
that entire failure class.
"""

from typing import Optional, Literal

from pydantic import BaseModel, Field


# A single invoice line item. Kept loose (all optional) because real invoices
# vary wildly — some have no unit price, some bundle tax into the line.
class LineItem(BaseModel):
    description: Optional[str] = Field(
        default=None, description="Line item description. If absent, null."
    )
    quantity: Optional[float] = Field(
        default=None, description="Quantity. If absent, null. Never fabricate."
    )
    unit_price: Optional[float] = Field(
        default=None, description="Unit price. If absent, null. Never fabricate."
    )
    amount: Optional[float] = Field(
        default=None, description="Line total. If absent, null. Never fabricate."
    )


class ExtractedInvoice(BaseModel):
    """The core extraction target for one document."""

    # ---- REQUIRED ----------------------------------------------------------
    # The only truly required field. If the model cannot find an invoice number,
    # that is an *unresolvable* error (the source genuinely lacks it) and the
    # document should go to human review rather than being retried forever.
    invoice_number: str = Field(
        description="The invoice number / ID. Required — every invoice has one."
    )

    # ---- ENUM + 'other' + detail ------------------------------------------
    # Closed category set with an escape hatch. The model labels confidently
    # within the set, or admits 'other' and explains, instead of forcing a
    # wrong label onto an unexpected document.
    doc_type: Literal["invoice", "credit_note", "receipt", "other"] = Field(
        description=(
            "Document type. Use 'invoice', 'credit_note', or 'receipt' when it "
            "clearly fits. Use 'other' for anything else and explain in "
            "doc_type_detail. Do NOT force a wrong label."
        )
    )
    doc_type_detail: Optional[str] = Field(
        default=None,
        description="If doc_type is 'other', describe the actual type here. Otherwise null.",
    )

    # ---- NULLABLE / OPTIONAL ----------------------------------------------
    # Information that may simply not be in the source. The rule, repeated in
    # every description, is: if absent return null, NEVER fabricate.
    vendor: Optional[str] = Field(
        default=None, description="Seller / vendor name. If absent, null. Never fabricate."
    )
    customer: Optional[str] = Field(
        default=None, description="Buyer / customer name. If absent, null. Never fabricate."
    )
    total_amount: Optional[float] = Field(
        default=None,
        description="Grand total as a number (no currency symbol). If absent, null. Never fabricate.",
    )
    currency: Optional[str] = Field(
        default=None,
        description="Currency code or symbol (e.g. USD, EUR, $). If absent, null. Never fabricate.",
    )
    invoice_date: Optional[str] = Field(
        default=None, description="Invoice issue date as written. If absent, null. Never fabricate."
    )
    due_date: Optional[str] = Field(
        default=None, description="Payment due date as written. If absent, null. Never fabricate."
    )
    line_items: Optional[list[LineItem]] = Field(
        default=None,
        description="List of line items. If none are itemized, null. Never fabricate rows.",
    )


# ---- Step 5: field-level confidence ---------------------------------------
class FieldWithConfidence(BaseModel):
    """One extracted value plus how confident the model is in it (0..1)."""

    value: Optional[str] = Field(
        default=None, description="The extracted value as a string, or null if absent."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in THIS field's value, 0.0 (guess) to 1.0 (certain).",
    )


class ExtractedInvoiceWithConfidence(BaseModel):
    """
    Same fields as ExtractedInvoice, but each is a value+confidence pair.

    Used by the review-routing layer (Step 5). Confidence is PER FIELD on
    purpose: one weak field (e.g. a smudged total) should not force the whole
    document to review if everything else is solid — and conversely, a
    confident-looking document can still hide one low-confidence field.
    """

    invoice_number: FieldWithConfidence
    doc_type: FieldWithConfidence
    vendor: FieldWithConfidence
    customer: FieldWithConfidence
    total_amount: FieldWithConfidence
    currency: FieldWithConfidence
    invoice_date: FieldWithConfidence
    due_date: FieldWithConfidence


# ---- The tool we force Claude to call --------------------------------------
# input_schema is derived straight from the Pydantic model, so the schema and
# the validator can never drift apart. The description hammers the no-fabricate
# rule one more time at the tool level, where the model weights it heavily.
extraction_tool = {
    "name": "extract_invoice",
    "description": (
        "Extract structured data from an invoice document. For any field whose "
        "information is absent from the source, return null — NEVER fabricate, "
        "guess, or infer a value that is not present. Required fields must come "
        "from the document. Use doc_type 'other' (with doc_type_detail) rather "
        "than forcing a wrong category."
    ),
    "input_schema": ExtractedInvoice.model_json_schema(),
}


# Confidence-variant tool definition, used by review_routing.py / Step 5.
extraction_tool_with_confidence = {
    "name": "extract_invoice_with_confidence",
    "description": (
        "Extract structured invoice data AND a per-field confidence score "
        "(0.0-1.0). For absent fields, set value to null and give an honest "
        "confidence. NEVER fabricate values. Be conservative: if you are "
        "guessing, the confidence must be low."
    ),
    "input_schema": ExtractedInvoiceWithConfidence.model_json_schema(),
}
