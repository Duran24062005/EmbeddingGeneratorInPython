import unittest
from types import SimpleNamespace

from main import MODEL, create_embedding


class FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def create(self, *, model: str, input: str) -> SimpleNamespace:
        self.calls.append({"model": model, "input": input})
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])],
        )


class FakeClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddings()


class CreateEmbeddingTests(unittest.TestCase):
    def test_sends_text_and_model_to_openai(self) -> None:
        client = FakeClient()

        response = create_embedding(client, "  Hola embeddings  ")

        self.assertEqual(response.data[0].embedding, [0.1, 0.2, 0.3])
        self.assertEqual(
            client.embeddings.calls,
            [{"model": MODEL, "input": "Hola embeddings"}],
        )

    def test_rejects_empty_text(self) -> None:
        with self.assertRaises(ValueError):
            create_embedding(FakeClient(), "   ")


if __name__ == "__main__":
    unittest.main()
