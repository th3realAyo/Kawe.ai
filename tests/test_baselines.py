import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from baselines import lead_summary, sparse_pagerank  # noqa: E402


class BaselineTests(unittest.TestCase):
    def test_sparse_pagerank_matches_networkx(self):
        import networkx as nx

        rng = np.random.default_rng(1)
        vocab = [f"w{i}" for i in range(30)]
        bags = [Counter(rng.choice(vocab, size=rng.integers(3, 12))) for _ in range(40)]
        bags.append(Counter({"isolated": 1}))  # dangling sentence

        matrix = np.zeros((len(bags), len(bags)))
        for i, a in enumerate(bags):
            for j, b in enumerate(bags):
                if i != j:
                    words = set(a) | set(b)
                    va = np.array([a[w] for w in words], float)
                    vb = np.array([b[w] for w in words], float)
                    matrix[i, j] = va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb))
        expected = nx.pagerank(nx.from_numpy_array(matrix))
        got = sparse_pagerank(bags)
        np.testing.assert_allclose(got, [expected[i] for i in range(len(bags))], atol=1e-6)

    def test_lead_takes_first_words(self):
        self.assertEqual(lead_summary("one two three four", 2), "one two")


if __name__ == "__main__":
    unittest.main()
