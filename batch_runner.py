"""
Exercise 3 — Step 4: Batch processing strategy.

Scale layer. Before writing any code, make the sync-vs-batch decision:

    Is someone WAITING on the result (a user, a blocking CI step)?
        YES -> synchronous API. Batch has no latency guarantee.
        NO  -> latency-tolerant (nightly / bulk)? -> Message Batches API:
               ~50% cheaper, results within 24h, matched by custom_id.

This module is the NO branch — 100 invoices processed overnight.

Two things the exam wants you to internalise:

  1. custom_id, not position.
     Batch results come back in ANY order. You match a result to its input by
     custom_id, never by index. submit_extraction_batch() therefore requires a
     stable id per document.

  2. Batch is one-shot; retry lives OUTSIDE the batch.
     A batch request is a single model turn. You CANNOT run the multi-turn
     validation-retry loop (Step 2) inside a batch — there is no place to send a
     follow-up mid-request. So the pattern is:
         batch -> collect -> for each FAILED custom_id, handle it OUTSIDE
         the batch (e.g. re-run through extract_with_retry, or chunk an
         oversized document and resubmit). You resubmit only the failures, never
         the whole batch.

SLA math (documented, per the brief):
    The batch window ceiling is 24h. If you owe a 30h SLA, you do NOT submit one
    100-doc batch and hope — you leave a safety margin. Submit in smaller
    batches on a schedule (e.g. ~4h cadence) so a single slow batch can't eat
    the whole SLA, and so failures surface early enough to reprocess before the
    deadline. Budget: 24h window + reprocessing time for failures + margin.
"""

import os

from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

from schema import extraction_tool

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

client = Anthropic()
MODEL = "claude-sonnet-4-6"


def submit_extraction_batch(documents: list[dict]):
    """
    Submit a batch of invoice extractions.

    documents: [{"id": <stable custom_id>, "text": <document text>}, ...]

    Each request forces the extract_invoice tool, exactly like the sync path,
    so the batch produces the same schema-shaped output. Returns the batch
    object; poll it with the SDK and then call collect_and_handle(batch.id).
    """
    requests = [
        Request(
            custom_id=doc["id"],  # how we re-match the (unordered) result later
            params=MessageCreateParamsNonStreaming(
                model=MODEL,
                max_tokens=2048,
                tools=[extraction_tool],
                tool_choice={"type": "tool", "name": "extract_invoice"},
                messages=[
                    {
                        "role": "user",
                        "content": f"Extract the invoice data from this document:\n\n{doc['text']}",
                    }
                ],
            ),
        )
        for doc in documents
    ]
    return client.messages.batches.create(requests=requests)


def collect_and_handle(batch_id: str) -> dict:
    """
    Collect batch results and triage them.

    Returns:
        {
          "results":  {custom_id: tool_input_dict},  # succeeded
          "failed":   [custom_id, ...],              # errored/canceled -> reprocess
          "oversized":[custom_id, ...],              # too large -> chunk & resubmit
        }

    Note the triage, NOT inline retry: a batch result can't be corrected in
    place. Failures are handed back to the caller to reprocess OUTSIDE the
    batch — re-run through extract_with_retry, or, for context-limit failures,
    chunk the document and resubmit only that custom_id in a new batch.
    """
    results: dict = {}
    failed: list[str] = []
    oversized: list[str] = []

    for item in client.messages.batches.results(batch_id):
        cid = item.custom_id
        rtype = item.result.type

        if rtype == "succeeded":
            msg = item.result.message
            tool_block = next(
                (b for b in msg.content if b.type == "tool_use"), None
            )
            if tool_block is not None:
                # Raw tool input — caller validates with ExtractedInvoice and,
                # on failure, reprocesses via the sync retry path.
                results[cid] = tool_block.input
            else:
                failed.append(cid)  # forced tool_choice should prevent this
        elif rtype == "errored":
            # Inspect the error. A context/size overflow means "chunk it";
            # anything else is a generic failure to reprocess.
            err_type = getattr(item.result.error, "type", "") or ""
            if "context" in err_type or "too_large" in err_type or "413" in err_type:
                oversized.append(cid)  # split into chunks, resubmit this id only
            else:
                failed.append(cid)
        else:
            # canceled / expired -> resubmit (only this custom_id).
            failed.append(cid)

    return {"results": results, "failed": failed, "oversized": oversized}
