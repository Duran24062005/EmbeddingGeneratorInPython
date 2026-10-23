"""Solicita un embedding a OpenAI y muestra la respuesta recibida."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

MODEL = "text-embedding-3-small"
DEFAULT_TEXT = "OpenAI transforma este texto en un vector numérico."


def create_embedding(client: Any, text: str, model: str = MODEL) -> Any:
    """Crea un embedding usando el cliente recibido.

    Recibir el cliente como argumento mantiene esta función fácil de probar
    sin realizar una llamada real a la API.
    """
    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("El texto no puede estar vacío.")

    return client.embeddings.create(model=model, input=normalized_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera un embedding de OpenAI a partir de un texto."
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Texto que se convertirá en embedding. Usa un texto de ejemplo si se omite.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Muestra la respuesta JSON completa, incluido el vector.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    args = build_parser().parse_args(argv)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Falta OPENAI_API_KEY. Cópiala en .env a partir de .env.example."
        )

    text = " ".join(args.text).strip() or DEFAULT_TEXT
    response = create_embedding(OpenAI(api_key=api_key), text)
    vector = response.data[0].embedding

    print(f"Texto: {text}")
    print(f"Modelo: {response.model}")
    print(f"Dimensiones del vector: {len(vector)}")
    print(f"Tokens usados: {response.usage.total_tokens}")
    print(f"Primeros 10 valores: {vector[:10]}")

    if args.full:
        print("\nRespuesta completa:")
        print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
