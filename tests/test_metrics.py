import unittest

from kawe.metrics import (
    MAX_UPLOAD_BYTES,
    TOP_N_MAX,
    TOP_N_MIN,
    build_summary_metrics,
    summary_top_n,
    validate_upload_size,
)


class ProductMetricsTests(unittest.TestCase):
    def test_summary_metrics_show_compression_and_listening_time(self):
        summary = "word " * 150

        metrics = build_summary_metrics(1_000, summary)

        self.assertEqual(metrics.source_words, 1_000)
        self.assertEqual(metrics.summary_words, 150)
        self.assertEqual(metrics.compression_percentage, 15)
        self.assertEqual(metrics.estimated_listening_minutes, 1)

    def test_empty_summary_has_no_listening_time(self):
        metrics = build_summary_metrics(500, "")

        self.assertEqual(metrics.summary_words, 0)
        self.assertEqual(metrics.compression_percentage, 0)
        self.assertEqual(metrics.estimated_listening_minutes, 0)

    def test_upload_limit_allows_exact_limit_and_rejects_larger_files(self):
        validate_upload_size(MAX_UPLOAD_BYTES)

        with self.assertRaises(ValueError):
            validate_upload_size(MAX_UPLOAD_BYTES + 1)

    def test_summary_top_n_scales_with_length_and_is_clamped(self):
        self.assertEqual(summary_top_n(30, 40), 12)  # 30% of 40 passages
        self.assertEqual(summary_top_n(10, 1), TOP_N_MIN)
        self.assertEqual(summary_top_n(80, 100), TOP_N_MAX)


if __name__ == "__main__":
    unittest.main()
