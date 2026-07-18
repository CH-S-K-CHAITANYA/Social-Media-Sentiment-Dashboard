import unittest
from types import SimpleNamespace

import pandas as pd

from src.xquik_source import load_xquik_posts


class FakeVectorizer:
    def transform(self, values):
        return values


class FakeModel:
    def predict(self, values):
        return [2]


class FakeSearch:
    def search(self, *, q, limit):
        self.query = q
        self.limit = limit
        tweet = {
            "id": "123",
            "text": "A great launch",
            "created_at": "2026-07-18T12:00:00Z",
            "like_count": 9,
            "retweet_count": 4,
        }
        return {"tweets": [tweet]}


class XquikSourceTest(unittest.TestCase):
    def test_normalizes_current_sdk_response(self):
        search = FakeSearch()
        client = SimpleNamespace(x=SimpleNamespace(tweets=search))

        result = load_xquik_posts(
            "launch",
            "Acme",
            25,
            FakeModel(),
            FakeVectorizer(),
            client,
        )

        self.assertEqual(search.query, "launch")
        self.assertEqual(search.limit, 25)
        self.assertEqual(result.iloc[0]["brand"], "Acme")
        self.assertEqual(result.iloc[0]["likes"], 9)
        self.assertEqual(result.iloc[0]["sentiment"], "positive")

    def test_preserves_missing_timestamp_as_not_available(self):
        search = FakeSearch()
        search.search = lambda **kwargs: {"tweets": [{"text": "A great launch"}]}
        client = SimpleNamespace(x=SimpleNamespace(tweets=search))

        result = load_xquik_posts(
            "launch",
            "Acme",
            25,
            FakeModel(),
            FakeVectorizer(),
            client,
        )

        self.assertTrue(pd.isna(result.iloc[0]["timestamp"]))


if __name__ == "__main__":
    unittest.main()
