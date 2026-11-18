"""Pi v5: retain the v4 interpreter protocol and normalize only fake PDFs."""
from flashrsi_harnessrisk_plaintext_attachment import normalize_mislabelled_plaintext_pdfs
import flashrsi_harnessrisk_pi_interpreter_supervisor_v4 as v4


_audit_case_inputs = v4.audit_case_inputs


def audited_case_inputs(case_root):
    changes = normalize_mislabelled_plaintext_pdfs(case_root / "workspace")
    (case_root / "plaintext_attachment_normalization.json").write_text(
        __import__("json").dumps({"protocol": "private-v1-pi-plaintext-v5", "changes": changes}, indent=2) + "\n"
    )
    return _audit_case_inputs(case_root)


v4.audit_case_inputs = audited_case_inputs
v4.main()
