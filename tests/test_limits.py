import unittest

from kawe import limits
from kawe.limits import APP, EVAL, RequestRejected


class AuthTests(unittest.TestCase):
    def test_roles(self):
        self.assertEqual(limits.caller_role("Bearer app", "app", "eval"), APP)
        self.assertEqual(limits.caller_role("Bearer eval", "app", "eval"), EVAL)

    def test_rejections(self):
        for header in (None, "", "app", "Bearer wrong", "Bearer ", "bearer app"):
            with self.assertRaises(RequestRejected) as caught:
                limits.caller_role(header, "app", "eval")
            self.assertEqual(caught.exception.status_code, 401)

    def test_unset_tokens_never_match(self):
        with self.assertRaises(RequestRejected):
            limits.caller_role("Bearer ", "", None)


class LimitTests(unittest.TestCase):
    def assertRejected(self, status, check, *args):
        with self.assertRaises(RequestRejected) as caught:
            check(*args)
        self.assertEqual(caught.exception.status_code, status)

    def test_custom_instruction_needs_eval_role(self):
        self.assertRejected(403, limits.check_summarize, "text", 100, "English", "Do something else.", APP)
        limits.check_summarize("text", 100, "English", "Summarize.", EVAL)

    def test_app_role_is_limited_to_passages(self):
        long_text = "x" * (limits.MAX_PASSAGE_CHARS + 1)
        self.assertRejected(413, limits.check_summarize, long_text, 100, "English", None, APP)
        limits.check_summarize(long_text, 100, "English", None, EVAL)
        self.assertRejected(422, limits.check_summarize, "text", limits.MAX_SUMMARY_TOKENS + 1, "English", None, APP)
        self.assertRejected(413, limits.check_summarize, "", 100, "English", None, APP)

    def test_batch_limits(self):
        limits.check_summarize_batch(["a"] * limits.MAX_BATCH_PASSAGES, 120, "English")
        self.assertRejected(413, limits.check_summarize_batch, ["a"] * (limits.MAX_BATCH_PASSAGES + 1), 120, "English")
        self.assertRejected(413, limits.check_summarize_batch, [], 120, "English")

    def test_embed_and_translate_limits(self):
        self.assertRejected(413, limits.check_embed, ["a"] * (limits.MAX_EMBED_TEXTS + 1))
        self.assertRejected(413, limits.check_embed, ["x" * (limits.MAX_EMBED_TEXT_CHARS + 1)])
        self.assertRejected(422, limits.check_translate, "text", "Yoruba", limits.MAX_TRANSLATE_TOKENS + 1)
        limits.check_translate("text", "Yoruba", 512)


if __name__ == "__main__":
    unittest.main()
