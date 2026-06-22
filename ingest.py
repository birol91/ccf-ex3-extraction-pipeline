"""
Exercise 3 — Ingestion: PDF (and plain-text) -> markdown string.

The extraction pipeline works on TEXT. This module turns a source document into
that text. For PDFs we use opendataloader-pdf; for .txt/.md we just read the
file directly (opendataloader only handles PDFs).

opendataloader-pdf API reality (verified, not invented):

  import opendataloader_pdf
  opendataloader_pdf.convert(input_path, output_dir=None, format=None,
                             to_stdout=False, ...) -> None

  convert() returns None — it does NOT return the text. With
  format="markdown" + output_dir it WRITES a .md file into output_dir. So the
  flow is: convert into a temp dir, find the produced .md, read it, return the
  string, clean up the temp dir. Requires Java 11+ (already installed).
"""

import os
import glob
import tempfile

import opendataloader_pdf


def _read_pdf(pdf_path: str) -> str:
    # Convert into a throwaway temp dir, then read back the markdown it wrote.
    with tempfile.TemporaryDirectory() as tmp:
        try:
            opendataloader_pdf.convert(
                input_path=[pdf_path],   # convert accepts a list of inputs
                output_dir=tmp,
                format="markdown",
            )
        except Exception as exc:  # most commonly: Java not found / Java < 11
            raise RuntimeError(
                f"opendataloader-pdf failed to convert {pdf_path!r}. "
                f"It requires Java 11+ on PATH. Underlying error: {exc}"
            ) from exc

        # convert() wrote one (or more) .md files somewhere under tmp. Find them.
        produced = glob.glob(os.path.join(tmp, "**", "*.md"), recursive=True)
        if not produced:
            raise RuntimeError(
                f"opendataloader-pdf produced no markdown for {pdf_path!r}. "
                f"Output dir was empty — the PDF may be unreadable or image-only."
            )

        # Concatenate in case it split output across files (usually just one).
        parts = []
        for md in sorted(produced):
            with open(md, "r", encoding="utf-8") as fh:
                parts.append(fh.read())
        return "\n\n".join(parts)


def pdf_to_text(pdf_path: str) -> str:
    """
    Return the text content of a document as a string.

    - .pdf  -> converted to markdown via opendataloader-pdf.
    - .txt / .md / anything else -> read directly as UTF-8 (already text).

    This lets the rest of the pipeline stay format-agnostic: it always gets a
    plain string regardless of how the document arrived.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    if pdf_path.lower().endswith(".pdf"):
        return _read_pdf(pdf_path)

    # Plain-text source — opendataloader is PDF-only, so just read it.
    with open(pdf_path, "r", encoding="utf-8") as fh:
        return fh.read()
