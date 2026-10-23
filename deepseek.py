"""Solicita una respuesta de texto a DeepSeek mediante su API compatible con OpenAI."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

MODEL = "deepseek-flash"
BASE_URL = "https://api.deepseek.com"
DEFAULT_PROMPT = "Explica en una frase qué es un embedding."


def ask_deepseek(client: Any, prompt: str, model: str = MODEL) -> Any:
    """Envía un prompt a DeepSeek y devuelve la respuesta completa."""
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        raise ValueError("El prompt no puede estar vacío.")

    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Responde de forma clara y breve."},
            {"role": "user", "content": normalized_prompt},
        ],
        stream=False,
        extra_body={"thinking": {"type": "enabled"}},
        reasoning_effort="high",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera una respuesta de texto usando DeepSeek."
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Pregunta o instrucción para DeepSeek.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Muestra la respuesta JSON completa.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    args = build_parser().parse_args(argv)

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit(
            "Falta DEEPSEEK_API_KEY. Agrégala al archivo .env, sin publicarla."
        )

    prompt = " ".join(args.prompt).strip() or DEFAULT_PROMPT
    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    response = ask_deepseek(client, prompt)
    answer = response.choices[0].message.content

    print(f"Prompt: {prompt}")
    print(f"Modelo: {response.model}")
    print(f"Tokens usados: {response.usage.total_tokens}")
    print(f"\nRespuesta:\n{answer}")

    if args.full:
        print("\nRespuesta completa:")
        print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
