"""Verify the PR convention gate independently of external services."""

import copy
import unittest

from check_pr import check


class PrConventionTests(unittest.TestCase):
    def setUp(self):
        self.pr = {"head": {"ref": "feat/SCRUM-38"}, "base": {"ref": "develop"},
                   "title": "SCRUM-38 sync", "body": "What I Did\nhttps://dajaba.atlassian.net/browse/SCRUM-38\nTo Reviewers"}

    def test_valid(self):
        self.assertEqual(check(self.pr), [])

    def test_wrong_branch_and_base(self):
        for part, ref in [("head", "feat/DAJ-38"), ("base", "main")]:
            pr = copy.deepcopy(self.pr)
            pr[part]["ref"] = ref
            self.assertTrue(check(pr))

    def test_missing_link_or_sections(self):
        for body in [None, "What I Did\nTo Reviewers", "https://dajaba.atlassian.net/browse/SCRUM-38"]:
            pr = copy.deepcopy(self.pr)
            pr["body"] = body
            self.assertTrue(check(pr))

    def test_key_prefix_does_not_match(self):
        pr = copy.deepcopy(self.pr)
        pr["title"] = "SCRUM-380"
        pr["body"] = "What I Did\nhttps://dajaba.atlassian.net/browse/SCRUM-380\nTo Reviewers"
        self.assertTrue(check(pr))

    def test_shell_text_is_only_data(self):
        self.pr["title"] += " $(echo unsafe) `echo unsafe`"
        self.assertEqual(check(self.pr), [])


if __name__ == "__main__":
    unittest.main()
