import re

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_MRN = re.compile(r"\b(?:MRN|Medical Record(?: Number)?)[:\s#-]*[A-Za-z0-9-]{4,}\b", re.I)
_PHONE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)"
)


def scrub_phi(text: str) -> str:
    """Mask contact identifiers before text is placed in the model prompt.

    Clinical facts stay intact. This only catches obvious emails, phone
    numbers, SSNs, and medical-record labels so a sample chart cannot leak
    them into the answer.
    """
    text = _EMAIL.sub("[REDACTED_EMAIL]", text)
    text = _SSN.sub("[REDACTED_SSN]", text)
    text = _MRN.sub("[REDACTED_MRN]", text)
    text = _PHONE.sub("[REDACTED_PHONE]", text)
    return text
