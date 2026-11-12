"""Build the stable, machine-readable evaluation report schema."""
from __future__ import annotations

from collections import Counter


SCHEMA_VERSION = 1


def build_report(sentences: list[dict]) -> dict:
    """Normalize sentence records and derive top-level issue indexes."""
    focus_counts = Counter()
    top_issues = []
    normalized = []
    for sentence in sentences:
        sentence_index = sentence['index']
        words = []
        for word_index, word in enumerate(sentence.get('words') or []):
            issues = word.get('issues') or []
            for issue_index, issue in enumerate(issues):
                if issue.get('category') == 'sound_change' and issue.get('expected'):
                    focus_counts[issue['expected']] += 1
                    if issue.get('heard'):
                        focus_counts[issue['heard']] += 1
                top_issues.append({'sentence': sentence_index, 'word': word_index,
                                   'issue': issue_index})
            words.append({
                'text': word['text'],
                'ipa': word.get('ipa'),
                'student_ipa': word.get('student_ipa'),
                'student_ms': word.get('student_ms'),
                'issues': issues,
            })
        normalized.append({
            'index': sentence_index,
            'text': sentence['text'],
            'missed': bool(sentence.get('missed')),
            'reference_ms': sentence.get('reference_ms'),
            'student_ms': sentence.get('student_ms'),
            'words': words,
            'hints': sentence.get('hints') or [],
        })

    return {
        'schema_version': SCHEMA_VERSION,
        'focus_sounds': [sound for sound, _ in sorted(
            focus_counts.items(), key=lambda item: -item[1])[:5]],
        'top_issues': top_issues[:10],
        'sentences': normalized,
    }
