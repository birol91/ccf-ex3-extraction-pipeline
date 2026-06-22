"""
Verification tests for Exercise 3 — structured data extraction pipeline.

Framework: stdlib unittest + unittest.mock (no pytest).

Constraint honored: NO real API calls. The Anthropic client objects created at
import time (extract.client, batch_runner.client) are replaced with mocks in the
tests that exercise the API paths, and client.messages.create /
client.messages.batches.* return hand-built fake responses. Importing
extract/batch_runner constructs Anthropic() with no API key, which the SDK
allows lazily (no network until a request is made), so import is safe.

Run:  python3 -m unittest test_pipeline -v
"""

import unittest
from unittest import mock

from pydantic import ValidationError

import schema
import extract
import review_routing
import batch_runner
from schema import (
    ExtractedInvoice,
    ExtractedInvoiceWithConfidence,
    FieldWithConfidence,
    extraction_tool,
)


# --------------------------------------------------------------------------- #
# Fake Anthropic response builders
# --------------------------------------------------------------------------- #
class FakeToolUseBlock:
    """Mimics a tool_use content block: .type == 'tool_use', .input, .id."""

    type = "tool_use"

    def __init__(self, tool_input, block_id="toolu_test"):
        self.input = tool_input
        self.id = block_id


class FakeTextBlock:
    type = "text"

    def __init__(self, text=""):
        self.text = text


class FakeResponse:
    """Mimics a Messages API response: has a .content list of blocks."""

    def __init__(self, blocks):
        self.content = blocks


def make_tool_response(tool_input, block_id="toolu_test"):
    """A response carrying a single tool_use block, as forced tool_choice yields."""
    return FakeResponse([FakeToolUseBlock(tool_input, block_id)])


# A fully-valid extraction payload (required fields present, others null).
VALID_INPUT = {
    "invoice_number": "INV-1",
    "doc_type": "invoice",
    "doc_type_detail": None,
    "vendor": "Acme Corp",
    "customer": None,
    "total_amount": 100.0,
    "currency": "USD",
    "invoice_date": None,
    "due_date": None,
    "line_items": None,
}


# =========================================================================== #
# 1. Schema / null behavior — fabrication prevention
# =========================================================================== #
class TestSchemaNullAndEnum(unittest.TestCase):
    def test_required_fields_are_invoice_number_and_doc_type(self):
        required = {
            name
            for name, f in ExtractedInvoice.model_fields.items()
            if f.is_required()
        }
        self.assertEqual(required, {"invoice_number", "doc_type"})

    def test_missing_required_invoice_number_rejected(self):
        with self.assertRaises(ValidationError):
            ExtractedInvoice(doc_type="invoice")  # no invoice_number

    def test_missing_required_doc_type_rejected(self):
        with self.assertRaises(ValidationError):
            ExtractedInvoice(invoice_number="X")  # no doc_type

    def test_nullable_fields_default_to_none(self):
        inv = ExtractedInvoice(invoice_number="X", doc_type="invoice")
        # Every optional field defaults to None — model returns null, not fabricated.
        for field in (
            "vendor",
            "customer",
            "total_amount",
            "currency",
            "invoice_date",
            "due_date",
            "line_items",
            "doc_type_detail",
        ):
            self.assertIsNone(getattr(inv, field), f"{field} should default to None")

    def test_doc_type_other_with_detail_accepted(self):
        inv = ExtractedInvoice(
            invoice_number="X",
            doc_type="other",
            doc_type_detail="purchase order",
        )
        self.assertEqual(inv.doc_type, "other")
        self.assertEqual(inv.doc_type_detail, "purchase order")

    def test_invalid_enum_value_rejected(self):
        with self.assertRaises(ValidationError):
            ExtractedInvoice(invoice_number="X", doc_type="banana")


# =========================================================================== #
# 2. classify_error — resolvable vs unresolvable (the exam's crux)
# =========================================================================== #
class TestClassifyError(unittest.TestCase):
    def _validation_error(self, **kwargs):
        try:
            ExtractedInvoice(**kwargs)
        except ValidationError as exc:
            return exc
        self.fail("Expected ValidationError but model validated")

    def test_format_error_is_resolvable(self):
        # required fields present, total_amount is a non-coercible string -> format
        exc = self._validation_error(
            invoice_number="X", doc_type="invoice", total_amount="not a number"
        )
        self.assertEqual(extract.classify_error(exc), "resolvable")

    def test_bad_enum_is_resolvable(self):
        # required present (invoice_number), doc_type enum wrong -> format mistake
        exc = self._validation_error(invoice_number="X", doc_type="banana")
        self.assertEqual(extract.classify_error(exc), "resolvable")

    def test_missing_required_is_unresolvable(self):
        # invoice_number absent -> info genuinely missing from source
        exc = self._validation_error(doc_type="invoice")
        self.assertEqual(extract.classify_error(exc), "unresolvable")

    def test_missing_required_dominates_mixed_errors(self):
        # one missing required (unresolvable) + one format error together ->
        # whole extraction is unresolvable.
        exc = self._validation_error(total_amount="not a number")  # no invoice_number, no doc_type
        self.assertEqual(extract.classify_error(exc), "unresolvable")

    def test_nested_lineitem_format_error_is_resolvable(self):
        # required present; a line item amount is non-coercible -> nested format
        exc = self._validation_error(
            invoice_number="X",
            doc_type="invoice",
            line_items=[{"amount": "not a number"}],
        )
        self.assertEqual(extract.classify_error(exc), "resolvable")


# =========================================================================== #
# 3. extract_with_retry — mocked client, retry paths
# =========================================================================== #
class TestExtractWithRetry(unittest.TestCase):
    def test_valid_first_pass_returns_invoice_one_call(self):
        fake_client = mock.MagicMock()
        fake_client.messages.create.return_value = make_tool_response(VALID_INPUT)

        with mock.patch.object(extract, "client", fake_client):
            result = extract.extract_with_retry("some document")

        self.assertIsInstance(result, ExtractedInvoice)
        self.assertEqual(result.invoice_number, "INV-1")
        self.assertEqual(fake_client.messages.create.call_count, 1)

    def test_tool_choice_is_forced(self):
        fake_client = mock.MagicMock()
        fake_client.messages.create.return_value = make_tool_response(VALID_INPUT)

        with mock.patch.object(extract, "client", fake_client):
            extract.extract_with_retry("doc")

        _, kwargs = fake_client.messages.create.call_args
        self.assertEqual(
            kwargs["tool_choice"], {"type": "tool", "name": "extract_invoice"}
        )
        # also confirm the extraction tool is the one passed
        self.assertEqual(kwargs["tools"], [extraction_tool])

    def test_format_error_then_fix_retries_and_succeeds(self):
        bad = dict(VALID_INPUT, total_amount="not a number")  # resolvable format error
        fake_client = mock.MagicMock()
        fake_client.messages.create.side_effect = [
            make_tool_response(bad),        # attempt 1: format error
            make_tool_response(VALID_INPUT),  # attempt 2: corrected
        ]

        with mock.patch.object(extract, "client", fake_client):
            result = extract.extract_with_retry("doc")

        self.assertIsInstance(result, ExtractedInvoice)
        self.assertEqual(fake_client.messages.create.call_count, 2)

    def test_resolvable_retry_followup_contains_tool_result_error(self):
        bad = dict(VALID_INPUT, total_amount="not a number")
        fake_client = mock.MagicMock()
        fake_client.messages.create.side_effect = [
            make_tool_response(bad),
            make_tool_response(VALID_INPUT),
        ]

        with mock.patch.object(extract, "client", fake_client):
            extract.extract_with_retry("doc")

        # Second call's messages must carry a tool_result with is_error=True.
        _, second_kwargs = fake_client.messages.create.call_args_list[1]
        messages = second_kwargs["messages"]
        tool_results = [
            block
            for msg in messages
            if isinstance(msg["content"], list)
            for block in msg["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        self.assertTrue(tool_results, "follow-up must include a tool_result block")
        self.assertTrue(tool_results[0]["is_error"])

    def test_missing_required_is_not_retried(self):
        # invoice_number missing -> unresolvable -> stop immediately, human review.
        missing = dict(VALID_INPUT)
        del missing["invoice_number"]
        fake_client = mock.MagicMock()
        fake_client.messages.create.return_value = make_tool_response(missing)

        with mock.patch.object(extract, "client", fake_client):
            result = extract.extract_with_retry("doc")

        self.assertIsNone(result)  # routed to human review
        self.assertEqual(
            fake_client.messages.create.call_count,
            1,
            "unresolvable error must NOT trigger a retry call",
        )

    def test_persistent_format_error_exhausts_retries(self):
        bad = dict(VALID_INPUT, total_amount="not a number")
        fake_client = mock.MagicMock()
        fake_client.messages.create.return_value = make_tool_response(bad)

        with mock.patch.object(extract, "client", fake_client):
            result = extract.extract_with_retry("doc", max_retries=2)

        self.assertIsNone(result)  # gave up to human review
        # max_retries=2 -> attempts 0,1,2 -> 3 calls total
        self.assertEqual(fake_client.messages.create.call_count, 3)


# =========================================================================== #
# 4. route_extraction — confidence routing
# =========================================================================== #
def _conf(value, confidence):
    return FieldWithConfidence(value=value, confidence=confidence)


def _all_high_confidence():
    return ExtractedInvoiceWithConfidence(
        invoice_number=_conf("INV-1", 0.95),
        doc_type=_conf("invoice", 0.99),
        vendor=_conf("Acme", 0.9),
        customer=_conf("Globex", 0.85),
        total_amount=_conf("100", 0.92),
        currency=_conf("USD", 0.99),
        invoice_date=_conf("2024-01-01", 0.8),
        due_date=_conf("2024-02-01", 0.8),
    )


class TestRouteExtraction(unittest.TestCase):
    def test_all_high_confidence_auto_accepts(self):
        decision, fields = review_routing.route_extraction(_all_high_confidence())
        self.assertEqual(decision, "auto_accept")
        self.assertEqual(fields, [])

    def test_low_confidence_field_routed_to_human(self):
        extracted = _all_high_confidence()
        extracted.total_amount = _conf("100", 0.3)  # below threshold (0.7)
        decision, fields = review_routing.route_extraction(extracted)
        self.assertEqual(decision, "human_review")
        self.assertEqual(fields, ["total_amount"])

    def test_only_low_fields_are_flagged(self):
        extracted = _all_high_confidence()
        extracted.vendor = _conf("Acme", 0.2)
        extracted.invoice_date = _conf("2024-01-01", 0.5)
        decision, fields = review_routing.route_extraction(extracted)
        self.assertEqual(decision, "human_review")
        # Only the two low-confidence fields, not the high-confidence ones.
        self.assertEqual(set(fields), {"vendor", "invoice_date"})
        self.assertNotIn("currency", fields)
        self.assertNotIn("invoice_number", fields)

    def test_value_exactly_at_threshold_is_accepted(self):
        # threshold check is strict "<", so 0.7 should NOT be flagged.
        extracted = _all_high_confidence()
        extracted.customer = _conf("Globex", review_routing.CONFIDENCE_THRESHOLD)
        decision, fields = review_routing.route_extraction(extracted)
        self.assertEqual(decision, "auto_accept")
        self.assertEqual(fields, [])


# =========================================================================== #
# 5. analyze_accuracy — doc_type x field breakdown (not aggregate)
# =========================================================================== #
class TestAnalyzeAccuracy(unittest.TestCase):
    def test_breakdown_is_per_doc_type_and_field(self):
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
        breakdown = review_routing.analyze_accuracy(results, truth)

        # Structure: keyed by doc_type, then field — NOT a single aggregate number.
        self.assertIn("invoice", breakdown)
        self.assertIn("receipt", breakdown)
        self.assertIn("total_amount", breakdown["invoice"])
        self.assertIn("invoice_number", breakdown["invoice"])

    def test_weak_cell_is_surfaced(self):
        # invoice/total_amount is the weak cell (1 of 2 correct = 50%);
        # invoice/invoice_number is perfect (100%). The breakdown must show this.
        results = [
            {"doc_type": "invoice", "invoice_number": "A1", "total_amount": "100"},
            {"doc_type": "invoice", "invoice_number": "A2", "total_amount": "999"},
        ]
        truth = [
            {"invoice_number": "A1", "total_amount": "100"},
            {"invoice_number": "A2", "total_amount": "250"},
        ]
        breakdown = review_routing.analyze_accuracy(results, truth)

        self.assertEqual(breakdown["invoice"]["total_amount"]["accuracy"], 0.5)
        self.assertEqual(breakdown["invoice"]["invoice_number"]["accuracy"], 1.0)
        # counts back the accuracy
        self.assertEqual(breakdown["invoice"]["total_amount"]["correct"], 1)
        self.assertEqual(breakdown["invoice"]["total_amount"]["total"], 2)

    def test_separate_doc_types_kept_distinct(self):
        results = [
            {"doc_type": "invoice", "total_amount": "100"},
            {"doc_type": "receipt", "total_amount": "100"},  # wrong
        ]
        truth = [
            {"total_amount": "100"},
            {"total_amount": "5"},
        ]
        breakdown = review_routing.analyze_accuracy(results, truth)
        self.assertEqual(breakdown["invoice"]["total_amount"]["accuracy"], 1.0)
        self.assertEqual(breakdown["receipt"]["total_amount"]["accuracy"], 0.0)


# =========================================================================== #
# 6. batch_runner — submit + collect/triage
# =========================================================================== #
class TestSubmitExtractionBatch(unittest.TestCase):
    def test_builds_one_request_per_doc_with_custom_id(self):
        fake_client = mock.MagicMock()
        fake_client.messages.batches.create.return_value = mock.MagicMock(id="batch_1")

        docs = [
            {"id": "doc-a", "text": "invoice A text"},
            {"id": "doc-b", "text": "invoice B text"},
        ]
        with mock.patch.object(batch_runner, "client", fake_client):
            batch_runner.submit_extraction_batch(docs)

        fake_client.messages.batches.create.assert_called_once()
        _, kwargs = fake_client.messages.batches.create.call_args
        requests = kwargs["requests"]
        self.assertEqual(len(requests), 2)

        custom_ids = [r["custom_id"] for r in requests]
        self.assertEqual(custom_ids, ["doc-a", "doc-b"])

        # Each request forces the extract_invoice tool, like the sync path.
        for r in requests:
            params = r["params"]
            self.assertEqual(
                params["tool_choice"], {"type": "tool", "name": "extract_invoice"}
            )
            self.assertEqual(params["tools"], [extraction_tool])


class FakeBatchResultMessage:
    def __init__(self, blocks):
        self.content = blocks


class FakeResultDetail:
    def __init__(self, rtype, message=None, error=None):
        self.type = rtype
        self.message = message
        self.error = error


class FakeBatchResultItem:
    def __init__(self, custom_id, result):
        self.custom_id = custom_id
        self.result = result


class TestCollectAndHandle(unittest.TestCase):
    def _run(self, items):
        fake_client = mock.MagicMock()
        fake_client.messages.batches.results.return_value = iter(items)
        with mock.patch.object(batch_runner, "client", fake_client):
            return batch_runner.collect_and_handle("batch_1")

    def test_succeeded_goes_to_results_keyed_by_custom_id(self):
        msg = FakeBatchResultMessage([FakeToolUseBlock(VALID_INPUT)])
        items = [
            FakeBatchResultItem("doc-a", FakeResultDetail("succeeded", message=msg)),
        ]
        out = self._run(items)
        self.assertIn("doc-a", out["results"])
        self.assertEqual(out["results"]["doc-a"], VALID_INPUT)
        self.assertEqual(out["failed"], [])
        self.assertEqual(out["oversized"], [])

    def test_errored_goes_to_failed(self):
        err = mock.MagicMock()
        err.type = "api_error"
        items = [
            FakeBatchResultItem("doc-b", FakeResultDetail("errored", error=err)),
        ]
        out = self._run(items)
        self.assertEqual(out["failed"], ["doc-b"])
        self.assertEqual(out["results"], {})

    def test_context_overflow_goes_to_oversized(self):
        err = mock.MagicMock()
        err.type = "context_length_exceeded"
        items = [
            FakeBatchResultItem("doc-c", FakeResultDetail("errored", error=err)),
        ]
        out = self._run(items)
        self.assertEqual(out["oversized"], ["doc-c"])
        self.assertEqual(out["failed"], [])

    def test_mixed_results_partitioned_by_custom_id(self):
        ok_msg = FakeBatchResultMessage([FakeToolUseBlock(VALID_INPUT, "t1")])
        big_err = mock.MagicMock()
        big_err.type = "too_large"
        generic_err = mock.MagicMock()
        generic_err.type = "api_error"
        items = [
            FakeBatchResultItem("ok-1", FakeResultDetail("succeeded", message=ok_msg)),
            FakeBatchResultItem("big-1", FakeResultDetail("errored", error=big_err)),
            FakeBatchResultItem("bad-1", FakeResultDetail("errored", error=generic_err)),
            FakeBatchResultItem("exp-1", FakeResultDetail("expired")),
        ]
        out = self._run(items)
        self.assertEqual(list(out["results"].keys()), ["ok-1"])
        self.assertEqual(out["oversized"], ["big-1"])
        self.assertEqual(set(out["failed"]), {"bad-1", "exp-1"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
