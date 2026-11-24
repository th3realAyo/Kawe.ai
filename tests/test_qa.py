import unittest
from unittest import mock

import numpy as np

from kawe import limits
from kawe.limits import RequestRejected
from kawe.modal_client import ModalServiceError
from kawe.qa import NOT_COVERED, QuestionError, answer_question, build_index, parse_answer, retrieve
from kawe.ranking import Chunk

# Three passages on separate axes; questions are embedded onto the axis they ask about.
TOPICS = ["maize yield rose", "fuel price fell", "school fees doubled"]
AXES = np.eye(4)


def index():
    chunks = [Chunk(text, f"Page {i + 1}", i, i) for i, text in enumerate(TOPICS)]
    return build_index(chunks, AXES[:3].tolist())


def embed_onto(axis_by_word):
    def embed(texts):
        out = []
        for text in texts:
            axis = next((a for word, a in axis_by_word.items() if word in text.lower()), 3)
            out.append(AXES[axis].tolist())
        return out

    return embed


EMBED = embed_onto({"maize": 0, "fuel": 1, "fees": 2})


class RetrievalTests(unittest.TestCase):
    def test_closest_passage_is_retrieved_and_unrelated_ones_dropped(self):
        found = retrieve(index(), AXES[1])
        self.assertEqual([p.chunk.text for p in found], ["fuel price fell"])
        self.assertEqual(found[0].number, 1)

    def test_nothing_relevant_returns_nothing(self):
        self.assertEqual(retrieve(index(), AXES[3]), [])

    def test_passages_are_numbered_in_document_order(self):
        mixed = np.array([0.2, 0.9, 0.6, 0.0])
        found = retrieve(index(), mixed / np.linalg.norm(mixed))
        self.assertEqual([p.chunk.chunk_index for p in found], [1, 2])
        self.assertEqual([p.number for p in found], [1, 2])

    def test_index_embeds_when_summary_had_no_vectors(self):
        chunks = [Chunk("a", "Page 1", 0, 0)]
        embed = mock.Mock(return_value=[[1.0, 0.0]])
        built = build_index(chunks, None, embed_fn=embed)
        embed.assert_called_once_with(["a"])
        self.assertEqual(built.vectors.shape, (1, 2))


class HybridRetrievalTests(unittest.TestCase):
    def test_keywords_pick_the_answer_when_embeddings_are_flat(self):
        texts = [
            "The introduction describes penalty shootouts in general.",
            "Related studies of shootouts in domestic cups.",
            "We find no first-mover advantage: the team kicking first does not win more often.",
            "Team strength from Elo ratings predicts shootout winners.",
        ]
        chunks = [Chunk(t, f"Page {i + 1}", i, i) for i, t in enumerate(texts)]
        flat = np.full((4, 4), 0.5) + np.eye(4) * 0.01  # every passage looks equally on-topic
        flat /= np.linalg.norm(flat, axis=1, keepdims=True)
        idx = build_index(chunks, flat.tolist())
        question_vector = flat[0]  # embeddings alone would rank the introduction first
        found = retrieve(idx, question_vector, "Does the team that kicks first win more often?", top_k=1)
        self.assertEqual(found[0].chunk.location_label, "Page 3")

    def test_keywords_cannot_rescue_off_topic_questions(self):
        found = retrieve(index(), AXES[3], "maize fuel fees")
        self.assertEqual(found, [])

    def test_stemming_matches_word_forms(self):
        from kawe.qa import tokenize

        self.assertEqual(tokenize("kicks kicking kicked"), ["kick", "kick", "kick"])
        self.assertNotIn("the", tokenize("the team"))


class ParseTests(unittest.TestCase):
    def setUp(self):
        self.retrieved = retrieve(index(), np.array([0.6, 0.6, 0.5, 0.0]))

    def test_citations_are_mapped_and_removed(self):
        text, cited, found = parse_answer("Yields rose [1]. Fuel got cheaper [2].", self.retrieved)
        self.assertTrue(found)
        self.assertEqual(text, "Yields rose. Fuel got cheaper.")
        self.assertEqual([p.number for p in cited], [1, 2])

    def test_not_found_and_unknown_citations(self):
        self.assertFalse(parse_answer("NOT_FOUND", self.retrieved)[2])
        self.assertFalse(parse_answer("  ", self.retrieved)[2])
        _, cited, _ = parse_answer("Something [9].", self.retrieved)
        self.assertEqual(len(cited), len(self.retrieved))  # uncited answers are attributed to all passages


class AnswerTests(unittest.TestCase):
    def test_english_question_is_answered_with_citation(self):
        answer_fn = mock.Mock(return_value="Maize yield rose [1].")
        answer = answer_question("What happened to maize?", index(), embed_fn=EMBED, answer_fn=answer_fn, translate_fn=mock.Mock())
        self.assertTrue(answer.found)
        self.assertEqual(answer.text, "Maize yield rose.")
        self.assertEqual([p.chunk.location_label for p in answer.cited], ["Page 1"])
        answer_fn.assert_called_once_with("What happened to maize?", ["maize yield rose"])

    def test_yoruba_question_is_translated_both_ways(self):
        def translate(text, language):
            return {"Kí ló ṣẹlẹ̀ sí owó epo?": "What happened to the fuel price?", "Fuel price fell.": "Owó epo dínkù."}[text] if language in ("English", "Yoruba") else text

        answer = answer_question(
            "Kí ló ṣẹlẹ̀ sí owó epo?", index(), "Yoruba", embed_fn=EMBED, answer_fn=lambda q, p: "Fuel price fell [1].", translate_fn=translate
        )
        self.assertEqual(answer.text, "Owó epo dínkù.")
        self.assertEqual(answer.english_text, "Fuel price fell.")
        self.assertEqual(answer.cited[0].chunk.text, "fuel price fell")

    def test_unrelated_question_says_the_document_does_not_say(self):
        answer_fn = mock.Mock()
        answer = answer_question("Who won the election?", index(), embed_fn=EMBED, answer_fn=answer_fn, translate_fn=mock.Mock())
        self.assertFalse(answer.found)
        self.assertEqual(answer.text, NOT_COVERED)
        answer_fn.assert_not_called()  # no passage was close enough to ask the model

    def test_model_refusal_says_the_document_does_not_say(self):
        answer = answer_question("Why did maize rise?", index(), embed_fn=EMBED, answer_fn=lambda q, p: "NOT_FOUND", translate_fn=mock.Mock())
        self.assertFalse(answer.found)
        self.assertEqual(answer.text, NOT_COVERED)

    def test_service_failures_and_bad_questions_raise_question_error(self):
        def down(q, p):
            raise ModalServiceError("down")

        with self.assertRaises(QuestionError):
            answer_question("What about maize?", index(), embed_fn=EMBED, answer_fn=down, translate_fn=mock.Mock())
        for bad in ("   ", "x" * 1001):
            with self.assertRaises(QuestionError):
                answer_question(bad, index(), embed_fn=EMBED, answer_fn=mock.Mock(), translate_fn=mock.Mock())


class AnswerLimitTests(unittest.TestCase):
    def test_limits(self):
        limits.check_answer("q?", ["p"] * limits.MAX_ANSWER_PASSAGES, 200)
        for args in (("", ["p"], 200), ("q", [], 200), ("q", ["p"] * 6, 200), ("q" * 1001, ["p"], 200), ("q", ["p"], 401)):
            with self.assertRaises(RequestRejected):
                limits.check_answer(*args)


if __name__ == "__main__":
    unittest.main()
