"""
Exercise 3 — Step 3: Few-shot examples for varied document formats.

The new idea here (not in the earlier briefs) is robustness to FORMAT variety.
The same invoice facts can be presented completely differently:

    - narrative prose ("Invoice 7781 was issued by ... on ...")
    - a labelled key/value block ("Invoice No: 7781")
    - a table / inline layout ("INV-99 | Widgets Inc | $400")

If you only ever show the model one layout, it overfits to that layout and
stumbles on the others. Few-shot examples that span DIFFERENT formats teach the
model the underlying task ("find these fields wherever they live") rather than a
surface pattern. That is the whole point of varying the example formats — not
just having examples, but having examples that look structurally unlike each
other.

Each example is a full turn triple:
    user document  ->  assistant tool_use  ->  user tool_result ("accepted")
This mirrors exactly the message shape the real extraction call uses, so the
model treats the examples as worked precedents for the same tool.

extract_with_fewshot() prepends these examples and then runs the same
validation-retry machinery from extract.py — few-shot improves the FIRST-pass
quality; the retry loop still backs it up.
"""

from extract import client, MODEL, classify_error, _route_to_human_review
from schema import ExtractedInvoice, extraction_tool
from pydantic import ValidationError


# Three examples in three deliberately different formats.
FEW_SHOT_EXAMPLES = [
    # --- Example 1: NARRATIVE prose -------------------------------------------
    {
        "role": "user",
        "content": (
            "Extract the invoice data from this document:\n\n"
            "Invoice number 7781 was issued by Northwind Traders to their "
            "client Contoso Ltd on the 5th of January 2024. The full amount of "
            "920 euros is payable by the end of the month."
        ),
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "fewshot_1",
                "name": "extract_invoice",
                "input": {
                    "invoice_number": "7781",
                    "doc_type": "invoice",
                    "doc_type_detail": None,
                    "vendor": "Northwind Traders",
                    "customer": "Contoso Ltd",
                    "total_amount": 920,
                    "currency": "EUR",
                    "invoice_date": "5 January 2024",
                    "due_date": "end of January 2024",
                    "line_items": None,
                },
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "fewshot_1", "content": "Accepted."}
        ],
    },
    # --- Example 2: LABELLED key/value block, with a MISSING field ------------
    # Note customer is absent -> the model returns null, NOT a guess. This
    # example doubles as a no-fabricate demonstration.
    {
        "role": "user",
        "content": (
            "Extract the invoice data from this document:\n\n"
            "Invoice No: INV-2024-555\n"
            "Seller: Globex Corporation\n"
            "Issued: 2024-02-14\n"
            "Amount Due: 3,400.00 USD\n"
        ),
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "fewshot_2",
                "name": "extract_invoice",
                "input": {
                    "invoice_number": "INV-2024-555",
                    "doc_type": "invoice",
                    "doc_type_detail": None,
                    "vendor": "Globex Corporation",
                    "customer": None,  # absent in source -> null, never invented
                    "total_amount": 3400.0,
                    "currency": "USD",
                    "invoice_date": "2024-02-14",
                    "due_date": None,
                    "line_items": None,
                },
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "fewshot_2", "content": "Accepted."}
        ],
    },
    # --- Example 3: TABLE / inline layout, with line items -------------------
    {
        "role": "user",
        "content": (
            "Extract the invoice data from this document:\n\n"
            "| Invoice | Vendor | Customer | Date | Total |\n"
            "| RCT-08 | QuickMart | Jane Doe | 2024-04-02 | $42.50 |\n"
            "Items: 2x Coffee @ $5.00; 1x Sandwich @ $32.50"
        ),
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "fewshot_3",
                "name": "extract_invoice",
                "input": {
                    "invoice_number": "RCT-08",
                    "doc_type": "receipt",
                    "doc_type_detail": None,
                    "vendor": "QuickMart",
                    "customer": "Jane Doe",
                    "total_amount": 42.5,
                    "currency": "USD",
                    "invoice_date": "2024-04-02",
                    "due_date": None,
                    "line_items": [
                        {"description": "Coffee", "quantity": 2, "unit_price": 5.0, "amount": 10.0},
                        {"description": "Sandwich", "quantity": 1, "unit_price": 32.5, "amount": 32.5},
                    ],
                },
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "fewshot_3", "content": "Accepted."}
        ],
    },
]


def extract_with_fewshot(
    document_text: str, max_retries: int = 2
) -> ExtractedInvoice | None:
    """
    Same contract as extract_with_retry, but primes the conversation with the
    varied-format few-shot examples first. Few-shot lifts first-pass accuracy on
    unusual layouts; the validation-retry loop still guards the result.
    """
    messages = list(FEW_SHOT_EXAMPLES) + [
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
            tool_choice={"type": "tool", "name": "extract_invoice"},
            messages=messages,
        )
        tool_block = next(b for b in resp.content if b.type == "tool_use")

        try:
            return ExtractedInvoice(**tool_block.input)
        except ValidationError as exc:
            if classify_error(exc) == "unresolvable":
                _route_to_human_review(
                    document_text, "unresolvable (info absent)", tool_block.input
                )
                return None
            if attempt == max_retries:
                _route_to_human_review(
                    document_text, "retries exhausted", tool_block.input
                )
                return None
            messages.append({"role": "assistant", "content": resp.content})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": (
                                f"Validation failed:\n{exc}\nFix only the named "
                                "fields. Leave truly-absent fields null; never fabricate."
                            ),
                            "is_error": True,
                        }
                    ],
                }
            )
    return None
