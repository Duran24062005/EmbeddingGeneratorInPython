import unittest
from types import SimpleNamespace

from deepseek import MODEL, ask_deepseek


class FakeChatCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Respuesta de prueba."),
                )
            ]
        )


class FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeChatCompletions())


class DeepSeekTests(unittest.TestCase):
    def test_sends_prompt_to_deepseek(self) -> None:
        client = FakeClient()

        response = ask_deepseek(client, "  Hola DeepSeek  ")

        self.assertEqual(response.choices[0].message.content, "Respuesta de prueba.")
        call = client.chat.completions.calls[0]
        self.assertEqual(call["model"], MODEL)
        self.assertEqual(call["stream"], False)
        self.assertEqual(
            call["messages"],
            [
                {"role": "system", "content": "Responde de forma clara y breve."},
                {"role": "user", "content": "Hola DeepSeek"},
            ],
        )

    def test_rejects_empty_prompt(self) -> None:
        with self.assertRaises(ValueError):
            ask_deepseek(FakeClient(), "   ")


if __name__ == "__main__":
    unittest.main()
