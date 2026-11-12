import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from structured_report import build_report


class StructuredReportTests(unittest.TestCase):
    def test_preserves_missing_data_and_builds_issue_references(self):
        report = build_report([
            {'index': 2, 'text': 'Thanks.', 'missed': False,
             'reference_ms': [100, 800], 'student_ms': [120, 780],
             'words': [{'text': 'Thanks', 'ipa': 'θ æ ŋ k s', 'student_ipa': 's æ ŋ k s',
                        'student_ms': [120, 780], 'issues': [
                            {'category': 'sound_change', 'expected': 'θ', 'heard': 's'}]}],
             'hints': []},
            {'index': 3, 'text': 'for you.', 'missed': True,
             'reference_ms': [900, 1400], 'student_ms': None,
             'words': [{'text': 'for', 'ipa': None, 'student_ipa': None,
                        'student_ms': None, 'issues': []}], 'hints': []},
        ])

        self.assertEqual(report['schema_version'], 1)
        self.assertEqual(report['focus_sounds'], ['θ', 's'])
        self.assertEqual(report['top_issues'], [{'sentence': 2, 'word': 0, 'issue': 0}])
        self.assertIsNone(report['sentences'][1]['student_ms'])
        self.assertIsNone(report['sentences'][1]['words'][0]['ipa'])
        self.assertTrue(report['sentences'][1]['missed'])

    def test_empty_issues_produce_empty_aggregates(self):
        report = build_report([{'index': 1, 'text': 'Hello.', 'words': [], 'hints': []}])
        self.assertEqual(report['focus_sounds'], [])
        self.assertEqual(report['top_issues'], [])
        self.assertEqual(report['sentences'][0]['words'], [])


if __name__ == '__main__':
    unittest.main()
