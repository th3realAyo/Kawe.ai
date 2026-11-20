import io
import unittest
from unittest import mock

import numpy as np

from kawe import modal_client
from kawe.extraction import ExtractedSegment, clean_page_text, extract_docx, strip_running_lines
from kawe.modal_client import ModalServiceError
from kawe.ranking import MAX_WORDS_PER_CHUNK, MIN_TAIL_WORDS, Chunk, centrality_scores, pagerank, rank_chunks, select_top, split_into_chunks
from kawe.summarizer import GroundedSummarizerError, clean_summary, generate_grounded_summary


def unit_vectors(n, dim=8, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, dim))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).tolist()


class ExtractionTests(unittest.TestCase):
    def test_clean_page_joins_hyphenated_breaks_and_whitespace(self):
        self.assertEqual(clean_page_text("large lan￾guage  models\r\n\r\nnext   line"), "large language models\nnext line")

    def test_running_headers_and_page_numbers_are_removed(self):
        topics = ["soil", "rain", "maize", "yield", "price", "trade"]
        pages = [
            f"Journal of Things, Vol 5\nThe {t} section opens.\nIt discusses {t} at length.\nThe {t} section ends.\n{100 + i}"
            for i, t in enumerate(topics)
        ]
        cleaned = strip_running_lines(pages)
        for i, (t, page) in enumerate(zip(topics, cleaned)):
            self.assertNotIn("Journal of Things", page)
            self.assertNotIn(str(100 + i), page)
            self.assertIn(f"The {t} section opens.", page)
            self.assertIn(f"The {t} section ends.", page)

    def test_short_documents_are_left_alone(self):
        pages = ["Header\nA\nB", "Header\nC\nD"]
        self.assertEqual(strip_running_lines(pages), pages)

    def test_docx_is_grouped_into_pseudo_pages(self):
        import docx

        document = docx.Document()
        for i in range(20):
            document.add_paragraph(" ".join(["word"] * 40) + f" p{i}")
        buffer = io.BytesIO()
        document.save(buffer)
        with mock.patch("docx.Document", return_value=docx.Document(io.BytesIO(buffer.getvalue()))):
            segments = extract_docx("ignored.docx")
        self.assertEqual([s.location_label for s in segments], ["Section 1", "Section 2", "Section 3"])
        self.assertIn("\n", segments[0].text)

    def docx_file(self, build):
        import docx

        document = docx.Document()
        build(document)
        buffer = io.BytesIO()
        document.save(buffer)
        return mock.patch("docx.Document", return_value=docx.Document(io.BytesIO(buffer.getvalue())))

    def test_docx_tables_are_kept_in_body_order(self):
        def build(document):
            document.add_paragraph("Before the table.")
            table = document.add_table(rows=2, cols=2)
            table.cell(0, 0).text, table.cell(0, 1).text = "Crop", "Yield"
            table.cell(1, 0).text, table.cell(1, 1).text = "Maize", "4.2 t/ha"
            document.add_paragraph("After the table.")

        with self.docx_file(build):
            text = extract_docx("ignored.docx")[0].text
        self.assertEqual(text.split("\n"), ["Before the table.", "Crop | Yield", "Maize | 4.2 t/ha", "After the table."])

    def test_docx_sections_follow_and_name_headings(self):
        def build(document):
            document.add_heading("Introduction", level=1)
            document.add_paragraph(" ".join(["intro"] * 150))
            document.add_heading("Methods", level=1)
            document.add_paragraph(" ".join(["method"] * 50))
            document.add_heading("Sampling", level=2)  # too little text yet to start a new section
            document.add_paragraph(" ".join(["sample"] * 20))

        with self.docx_file(build):
            segments = extract_docx("ignored.docx")
        self.assertEqual([s.location_label for s in segments], ['Section 1 ("Introduction")', 'Section 2 ("Methods")'])
        self.assertTrue(segments[1].text.startswith("Methods"))
        self.assertIn("Sampling", segments[1].text)


class RankingTests(unittest.TestCase):
    def test_long_segments_are_split_and_keep_their_location(self):
        segments = [ExtractedSegment(" ".join(["w"] * (MAX_WORDS_PER_CHUNK * 2 + 5)), "Page 1", 0), ExtractedSegment("short", "Page 2", 1)]
        chunks = split_into_chunks(segments)
        # 365 unpunctuated words: cut at 180, and the 5-word tail joins the piece before it.
        self.assertEqual([c.location_label for c in chunks], ["Page 1"] * 2 + ["Page 2"])
        self.assertEqual([len(c.text.split()) for c in chunks], [180, 185, 1])
        self.assertEqual([c.chunk_index for c in chunks], [0, 1, 2])

    def test_chunks_end_at_sentence_boundaries(self):
        sentence = " ".join(["word"] * 29) + " end."  # 30 words
        text = " ".join([sentence] * 14)  # 420 words
        chunks = split_into_chunks([ExtractedSegment(text, "Page 1", 0)])
        self.assertEqual([len(c.text.split()) for c in chunks], [180, 180, 60])
        for chunk in chunks:
            self.assertTrue(chunk.text.endswith("end."))
            self.assertTrue(chunk.text.startswith("word"))

    def test_short_tail_is_merged(self):
        sentence = " ".join(["word"] * 29) + " end."
        text = " ".join([sentence] * 6) + " " + " ".join(["tail"] * (MIN_TAIL_WORDS - 1)) + "."
        chunks = split_into_chunks([ExtractedSegment(text, "Page 1", 0)])
        self.assertEqual(len(chunks), 1)

    def test_pagerank_matches_networkx(self):
        import networkx as nx

        rng = np.random.default_rng(3)
        weights = np.clip(rng.normal(size=(25, 25)), 0, None)
        weights = (weights + weights.T) / 2
        np.fill_diagonal(weights, 0)
        weights[4, :] = weights[:, 4] = 0  # dangling node
        expected = nx.pagerank(nx.from_numpy_array(weights))
        np.testing.assert_allclose(pagerank(weights), [expected[i] for i in range(25)], atol=1e-6)

    def test_mmr_skips_near_duplicates(self):
        chunks = [Chunk(f"c{i}", f"Page {i}", i, i) for i in range(4)]
        vectors = np.array([[1, 0, 0], [1, 0.01, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        scores = [0.9, 0.89, 0.5, 0.1]
        self.assertEqual([c.chunk_index for c in select_top(chunks, scores, 2)], [0, 1])
        self.assertEqual([c.chunk_index for c in select_top(chunks, scores, 2, vectors.tolist())], [0, 2])

    def test_central_passage_scores_highest(self):
        hub = np.ones(4) / 2
        vectors = [hub, [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        scores = centrality_scores(vectors)
        self.assertEqual(int(np.argmax(scores)), 0)

    def test_selection_returns_document_order(self):
        chunks = [Chunk(f"c{i}", f"Page {i}", i, i) for i in range(5)]
        picked = select_top(chunks, [0.1, 0.9, 0.2, 0.8, 0.3], top_n=2)
        self.assertEqual([c.chunk_index for c in picked], [1, 3])

    def test_no_embedding_call_when_every_chunk_fits(self):
        chunks = [Chunk("a", "Page 2", 1, 0), Chunk("b", "Page 1", 0, 1)]
        embed = mock.Mock()
        self.assertEqual([c.text for c in rank_chunks(chunks, top_n=5, embed_fn=embed)], ["b", "a"])
        embed.assert_not_called()


class SummarizerTests(unittest.TestCase):
    def segments(self, n=10):
        return [ExtractedSegment(f"Passage number {i} " + "text " * 50, f"Page {i + 1}", i) for i in range(n)]

    def test_preamble_and_quotes_are_removed(self):
        self.assertEqual(clean_summary('Here is a summary of the passage in 2 sentences:\n\n"The study found X."'), "The study found X.")
        self.assertEqual(clean_summary("The study found X."), "The study found X.")

    def test_summary_keeps_citations_in_document_order(self):
        result = generate_grounded_summary(
            self.segments(),
            top_n=3,
            embed_fn=lambda texts: unit_vectors(len(texts)),
            summarize_fn=lambda text, lang: "Summary of " + text.split()[2] + ".",
        )
        self.assertEqual(len(result.sections), 3)
        orders = [s.order for s in result.sections]
        self.assertEqual(orders, sorted(orders))
        for section in result.sections:
            self.assertIn(section.summary_text.split()[2].rstrip("."), section.source_excerpt)
        self.assertEqual(result.total_chunks, 10)
        self.assertEqual(set(result.timings), {"embed", "rank", "rewrite"})

    def test_failed_passages_are_skipped(self):
        calls = {"n": 0}

        def flaky(text, lang):
            calls["n"] += 1
            if "number 0 " in text:
                raise ModalServiceError("boom")
            return "ok"

        with mock.patch("kawe.summarizer.time.sleep"):
            result = generate_grounded_summary(self.segments(3), top_n=6, summarize_fn=flaky)
        self.assertEqual(result.skipped_chunks, 1)
        self.assertEqual(len(result.sections), 2)
        self.assertEqual(calls["n"], 4)  # one retry for the failing passage

    def test_default_path_rewrites_in_one_batch(self):
        batch = mock.Mock(side_effect=lambda texts, lang: [f"Summary {i}." for i in range(len(texts))])
        with mock.patch("kawe.summarizer._default_summarize_batch", batch):
            result = generate_grounded_summary(self.segments(3), top_n=6)
        batch.assert_called_once()
        self.assertEqual([s.summary_text for s in result.sections], ["Summary 0.", "Summary 1.", "Summary 2."])

    def test_failed_batch_falls_back_to_single_passages(self):
        batch = mock.Mock(side_effect=ModalServiceError("batch down"))
        single = mock.Mock(return_value="ok")
        with mock.patch("kawe.summarizer._default_summarize_batch", batch), mock.patch("kawe.summarizer._default_summarize", single):
            result = generate_grounded_summary(self.segments(3), top_n=6)
        self.assertEqual(single.call_count, 3)
        self.assertEqual(len(result.sections), 3)

    def test_errors_a_retry_cannot_fix_are_not_retried(self):
        calls = {"n": 0}

        def unauthorized(text, lang):
            calls["n"] += 1
            raise ModalServiceError("401", retryable=False)

        with mock.patch("kawe.summarizer.time.sleep") as sleep:
            with self.assertRaises(GroundedSummarizerError):
                generate_grounded_summary(self.segments(2), top_n=6, summarize_fn=unauthorized)
        self.assertEqual(calls["n"], 2)
        sleep.assert_not_called()

    def test_embedding_failure_is_reported(self):
        def broken(texts):
            raise ModalServiceError("down")

        with self.assertRaises(GroundedSummarizerError):
            generate_grounded_summary(self.segments(), top_n=2, embed_fn=broken, summarize_fn=lambda t, l: "x")


class ModalClientTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict("os.environ", {"MODAL_SUMMARIZE_URL": "https://example.test", "RAG_AUTH_TOKEN": "t"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def response(self, payload, status=200):
        return mock.Mock(status_code=status, json=lambda: payload, text=str(payload))

    def test_instruction_requires_a_redeployed_endpoint(self):
        eval_token = mock.patch.dict("os.environ", {"EVAL_AUTH_TOKEN": "e"})
        eval_token.start()
        self.addCleanup(eval_token.stop)
        with mock.patch("requests.post", return_value=self.response({"summary": "s"})):
            with self.assertRaises(ModalServiceError):
                modal_client.summarize("text", instruction="Summarize the document.")
        with mock.patch("requests.post", return_value=self.response({"summary": "s", "instruction_used": True})):
            self.assertEqual(modal_client.summarize("text", instruction="Summarize the document."), "s")

    def test_instruction_is_sent_with_the_eval_token(self):
        with mock.patch.dict("os.environ", {"EVAL_AUTH_TOKEN": "e"}):
            with mock.patch("requests.post", return_value=self.response({"summary": "s", "instruction_used": True})) as post:
                modal_client.summarize("text", instruction="Summarize.")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer e")

    def test_instruction_without_eval_token_fails_early(self):
        with mock.patch("requests.post") as post, self.assertRaises(ModalServiceError) as caught:
            modal_client.summarize("text", instruction="Summarize.")
        post.assert_not_called()
        self.assertFalse(caught.exception.retryable)

    def test_http_errors_become_service_errors(self):
        with mock.patch("requests.post", return_value=self.response({"detail": "no"}, status=401)):
            with self.assertRaises(ModalServiceError) as caught:
                modal_client.summarize("text")
        self.assertFalse(caught.exception.retryable)
        with mock.patch("requests.post", return_value=self.response({"detail": "busy"}, status=503)):
            with self.assertRaises(ModalServiceError) as caught:
                modal_client.summarize("text")
        self.assertTrue(caught.exception.retryable)

    def test_error_messages_never_echo_the_request(self):
        echoed = {"detail": [{"type": "string_type", "input": "CONFIDENTIAL passage text"}]}
        with mock.patch("requests.post", return_value=self.response(echoed, status=422)):
            with self.assertRaises(ModalServiceError) as caught:
                modal_client.summarize("CONFIDENTIAL passage text")
        self.assertNotIn("CONFIDENTIAL", str(caught.exception))
        with mock.patch("requests.post", return_value=self.response({"detail": "Text too long."}, status=413)):
            with self.assertRaises(ModalServiceError) as caught:
                modal_client.summarize("text")
        self.assertIn("Text too long.", str(caught.exception))

    def test_malformed_responses_become_service_errors(self):
        not_json = mock.Mock(status_code=200, text="<html>", json=mock.Mock(side_effect=ValueError("no json")))
        for response in (not_json, self.response({"unexpected": 1}), self.response(["a list"])):
            with mock.patch("requests.post", return_value=response):
                with self.assertRaises(ModalServiceError):
                    modal_client.summarize("text")

    def test_batch_must_return_one_summary_per_passage(self):
        with mock.patch.dict("os.environ", {"MODAL_SUMMARIZE_BATCH_URL": "https://example.test"}):
            with mock.patch("requests.post", return_value=self.response({"summaries": ["a", "b"]})):
                self.assertEqual(modal_client.summarize_batch(["x", "y"]), ["a", "b"])
            with mock.patch("requests.post", return_value=self.response({"summaries": ["a"]})):
                with self.assertRaises(ModalServiceError):
                    modal_client.summarize_batch(["x", "y"])

    def test_translation_budget_grows_with_length_within_limits(self):
        self.assertEqual(modal_client.translation_token_budget("one two"), 128)
        self.assertEqual(modal_client.translation_token_budget(" ".join(["w"] * 100)), 664)
        self.assertEqual(modal_client.translation_token_budget(" ".join(["w"] * 1000)), 1024)


if __name__ == "__main__":
    unittest.main()
