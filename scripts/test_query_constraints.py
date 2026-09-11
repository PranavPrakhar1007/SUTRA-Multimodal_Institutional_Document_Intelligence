"""Focused tests for domain-neutral query constraint handling."""

import sys
import unittest
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.query_constraints import constrain_candidates, extract_constraints


class TestQueryConstraints(unittest.TestCase):
    def test_extracts_multiple_explicit_constraints(self):
        constraints = extract_constraints("which subject is at 4-4:55 pm on thursday for btech 5th sem")
        self.assertEqual(constraints.semester, "5")
        self.assertEqual(constraints.weekdays, ("thursday",))
        self.assertEqual(constraints.times, ("4-4:55 pm",))

    def test_prefers_candidate_matching_explicit_constraints(self):
        candidates = [
            {"notice_id": "doc", "page_number": 1, "text": "B.Tech 7th Semester Thursday 4-4:55 PM"},
            {"notice_id": "doc", "page_number": 2, "text": "B.Tech 5th Semester Thursday 4-4:55 PM"},
        ]
        selected, constraints = constrain_candidates(
            candidates,
            "find the 5th semester document",
        )
        self.assertTrue(constraints.has_explicit_constraints)
        self.assertEqual([item["page_number"] for item in selected], [2])

    def test_does_not_drop_candidates_when_ocr_has_no_constraints(self):
        candidates = [{"notice_id": "doc", "page_number": 1, "text": "unreadable scan"}]
        selected, _ = constrain_candidates(candidates, "find the value for 5th semester")
        self.assertEqual(selected, candidates)

    def test_recognizes_structured_lookup_queries(self):
        constraints = extract_constraints("which subject is at 4-4:55 pm on thursday for btech 5th sem")
        self.assertTrue(constraints.is_structured_lookup)


if __name__ == "__main__":
    unittest.main()