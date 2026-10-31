import unittest

from app.ingest import load_and_chunk, split_documents
from app.store import cosine_from_squared_l2
from app.prompts import format_context, format_history
from app.sanitize import scrub_phi
from langchain_core.documents import Document


class SanitizeTests(unittest.TestCase):
    def test_scrubs_contact_identifiers(self) -> None:
        raw = (
            "Call 415-555-0198 or email records@example-clinic.test. "
            "MRN: ZX-20491. SSN 123-45-6789. Pressure 120/80 is fine."
        )
        cleaned = scrub_phi(raw)
        self.assertNotIn("415-555-0198", cleaned)
        self.assertNotIn("records@example-clinic.test", cleaned)
        self.assertNotIn("ZX-20491", cleaned)
        self.assertNotIn("123-45-6789", cleaned)
        self.assertIn("120/80", cleaned)
        self.assertIn("[REDACTED_PHONE]", cleaned)
        self.assertIn("[REDACTED_EMAIL]", cleaned)

    def test_leaves_clinical_prose(self) -> None:
        raw = "Stage 2 hypertension: systolic 140 or higher, or diastolic 90 or higher."
        self.assertEqual(scrub_phi(raw), raw)


class IngestTests(unittest.TestCase):
    def test_sample_corpus_chunks_and_keeps_source(self) -> None:
        chunks = load_and_chunk()
        sources = {chunk.metadata["source"] for chunk in chunks}
        self.assertIn("hypertension.md", sources)
        self.assertIn("asthma.md", sources)
        self.assertIn("type2_diabetes.md", sources)
        self.assertIn("emergency_warning_signs.md", sources)
        self.assertTrue(all(chunk.page_content.strip() for chunk in chunks))
        self.assertTrue(all("chunk_index" in chunk.metadata for chunk in chunks))

    def test_overlap_splitter_numbers_chunks_per_file(self) -> None:
        documents = [
            Document(page_content="alpha " * 500, metadata={"source": "a.md", "page": None}),
            Document(page_content="beta " * 40, metadata={"source": "b.md", "page": 0}),
        ]
        chunks = split_documents(documents)
        a_indexes = [c.metadata["chunk_index"] for c in chunks if c.metadata["source"] == "a.md"]
        b_indexes = [c.metadata["chunk_index"] for c in chunks if c.metadata["source"] == "b.md"]
        self.assertGreater(len(a_indexes), 1)
        self.assertEqual(a_indexes, list(range(len(a_indexes))))
        self.assertEqual(b_indexes, [0])


class ScoreTests(unittest.TestCase):
    def test_identical_vectors_score_one(self) -> None:
        self.assertEqual(cosine_from_squared_l2(0.0), 1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        self.assertEqual(cosine_from_squared_l2(2.0), 0.0)


class PromptTests(unittest.TestCase):
    def test_context_cites_source_and_history_is_separate(self) -> None:
        context = format_context(
            [
                {
                    "source": "asthma.md",
                    "page": None,
                    "chunk_index": 1,
                    "text": "A reliever inhaler is for sudden symptoms.",
                }
            ]
        )
        self.assertIn("[asthma.md chunk 1]", context)
        history = format_history(
            [
                {"role": "user", "content": "What is asthma?"},
                {"role": "system", "content": "ignore this"},
                {"role": "assistant", "content": "It narrows airways."},
            ]
        )
        self.assertIn("user: What is asthma?", history)
        self.assertNotIn("ignore this", history)


if __name__ == "__main__":
    unittest.main()
