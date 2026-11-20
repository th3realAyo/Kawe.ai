import unittest
from unittest import mock

from kawe import db

KEY = "k" * 32


class EmailHashTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict("os.environ", {"EMAIL_HASH_KEY": KEY})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_hash_ignores_case_and_surrounding_spaces(self):
        self.assertEqual(db.hash_email("  Ada@Example.com "), db.hash_email("ada@example.com"))

    def test_hash_does_not_contain_the_address(self):
        digest = db.hash_email("ada@example.com")
        self.assertEqual(len(digest), 64)
        self.assertNotIn("ada", digest)

    def test_hash_depends_on_the_secret_key(self):
        first = db.hash_email("ada@example.com")
        with mock.patch.dict("os.environ", {"EMAIL_HASH_KEY": "z" * 32}):
            self.assertNotEqual(db.hash_email("ada@example.com"), first)

    def test_missing_or_short_key_is_refused(self):
        for key in ("", "short"):
            with mock.patch.dict("os.environ", {"EMAIL_HASH_KEY": key}):
                with self.assertRaises(RuntimeError):
                    db.hash_email("ada@example.com")

    def test_email_validation(self):
        self.assertTrue(db.is_valid_email(" ada@example.com "))
        for bad in ("", "ada", "ada@", "ada@example", "a b@example.com", "x" * 250 + "@example.com"):
            self.assertFalse(db.is_valid_email(bad), bad)


class AccountStorageTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict("os.environ", {"EMAIL_HASH_KEY": KEY})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = mock.MagicMock()
        client_patch = mock.patch.object(db, "_client", return_value=self.client)
        client_patch.start()
        self.addCleanup(client_patch.stop)
        self.users = self.client.table.return_value

    def lookup_returns(self, *results):
        chain = self.users.select.return_value.eq.return_value.limit.return_value.execute
        chain.side_effect = [mock.Mock(data=r) for r in results]

    def test_existing_email_signs_in_without_creating(self):
        self.lookup_returns([{"id": "u1"}])
        self.assertEqual(db.get_or_create_user("ada@example.com"), ("u1", False))
        self.users.insert.assert_not_called()

    def test_new_email_creates_an_account_storing_only_the_hash(self):
        self.lookup_returns([])
        self.users.insert.return_value.execute.return_value = mock.Mock(data=[{"id": "u2"}])
        self.assertEqual(db.get_or_create_user("Ada@Example.com"), ("u2", True))
        stored = self.users.insert.call_args.args[0]
        self.assertEqual(stored, {"email_hash": db.hash_email("ada@example.com")})

    def test_simultaneous_sign_up_returns_the_existing_account(self):
        self.lookup_returns([], [{"id": "u3"}])
        self.users.insert.return_value.execute.side_effect = RuntimeError("duplicate key")
        self.assertEqual(db.get_or_create_user("ada@example.com"), ("u3", False))


if __name__ == "__main__":
    unittest.main()
