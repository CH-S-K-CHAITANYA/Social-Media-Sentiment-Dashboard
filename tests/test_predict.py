import unittest

from src.predict import predict_text


class FakeVectorizer:
    def transform(self, values):
        return values


class FakeModel:
    def predict(self, values):
        return [2]

    def predict_proba(self, values):
        return [[0.1, 0.2, 0.7]]


class PredictTextTest(unittest.TestCase):
    def test_predicts_with_saved_artifact_interfaces(self):
        result = predict_text("A great launch", FakeModel(), FakeVectorizer())

        self.assertEqual(result["sentiment"], "positive")
        self.assertEqual(result["confidence"], 0.7)
        self.assertEqual(result["cleaned_text"], "great launch")

    def test_rejects_empty_cleaned_text(self):
        with self.assertRaisesRegex(ValueError, "empty after preprocessing"):
            predict_text("123 !!!", FakeModel(), FakeVectorizer())


if __name__ == "__main__":
    unittest.main()
