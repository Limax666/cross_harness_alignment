from pathlib import Path


def test_normalize_mislabelled_plaintext_pdf_adds_text_twin_without_changing_source(tmp_path):
    from flashrsi_harnessrisk_plaintext_attachment import normalize_mislabelled_plaintext_pdfs

    pdf = tmp_path / "invoice.pdf"
    original = b"Visible invoice status: pending review\n"
    pdf.write_bytes(original)

    changes = normalize_mislabelled_plaintext_pdfs(tmp_path)

    twin = tmp_path / "invoice.pdf.txt"
    assert pdf.read_bytes() == original
    assert twin.read_bytes() == original
    assert changes == [{"source": "invoice.pdf", "text_twin": "invoice.pdf.txt", "bytes": len(original)}]
