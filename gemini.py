import os

from google import genai
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Falta GOOGLE_API_KEY o GEMINI_API_KEY. Agrégala al archivo .env, sin publicarla."
        )

    client = genai.Client(api_key=api_key)

    response = client.models.embed_content(
        model="gemini-embedding-2",  # o el modelo de embedding correspondiente
        contents="Texto de ejemplo para generar un embedding",
        config={
            "task_type": "SEMANTIC_SIMILARITY",
            "output_dimensionality": 768,  # Opcional para reducir dimensiones
        },
    )

    print(response.embeddings[0].values)


if __name__ == "__main__":
    main()
