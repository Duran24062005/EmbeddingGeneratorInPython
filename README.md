# OpenAI embeddings y DeepSeek con `uv`

Proyecto didáctico mínimo con dos scripts independientes:

- `main.py`: envía texto a OpenAI y muestra el vector embedding.
- `deepseek.py`: envía un prompt a DeepSeek y muestra una respuesta de texto.

## 1. Preparar el entorno

Requisitos:

- Python 3.10 o superior.
- [`uv`](https://docs.astral.sh/uv/).
- Una clave de API de OpenAI.

Instala las dependencias con:

```bash
uv sync
```

Crea el archivo local de variables de entorno:

```bash
cp .env.example .env
```

Después edita `.env` y coloca tu clave:

```dotenv
OPENAI_API_KEY=tu_clave_aqui
DEEPSEEK_API_KEY=tu_clave_deepseek_aqui
```

`.env` está ignorado por Git para no guardar la clave en el repositorio.

No pegues claves en el código, commits, capturas ni chats. Si una clave se
comparte accidentalmente, revócala en el panel del proveedor y crea una nueva
antes de continuar.

## 2. Ejecutar el ejemplo

Sin argumentos se usa un texto de ejemplo:

```bash
uv run main.py
```

También puedes enviar tu propio texto:

```bash
uv run main.py "Python es un lenguaje fácil de aprender"
```

La salida muestra el modelo, la cantidad de dimensiones, los tokens usados y
los primeros 10 números del vector. Para observar la respuesta JSON completa,
incluido todo el vector:

```bash
uv run main.py --full "Python es un lenguaje fácil de aprender"
```

## 3. Ejecutar DeepSeek

DeepSeek usa un endpoint de chat compatible con el SDK de OpenAI. El script
configura el cliente con `https://api.deepseek.com` y usa el modelo
`deepseek-flash`:

```bash
uv run deepseek.py "Explica qué es un embedding en una frase"
```

Para observar la respuesta JSON completa:

```bash
uv run deepseek.py --full "Explica qué es un embedding en una frase"
```

DeepSeek no se usa aquí para generar embeddings: su API oficial documenta
`/chat/completions`, mientras que el embedding del proyecto sigue siendo el de
OpenAI. Así se evita presentar como equivalentes dos capacidades distintas.

## 4. Qué devuelve la API de OpenAI

La llamada importante está en `main.py`:

```python
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Python es un lenguaje fácil de aprender",
)

vector = response.data[0].embedding
```

`vector` es una lista de números decimales. El programa imprime su longitud y
algunos valores para que la terminal siga siendo legible. El vector completo
está disponible en `response.data[0].embedding`.

La imagen usa `text-embedding-ada-002`, que es un modelo anterior. Para
reproducir ese identificador de forma explícita, cambia la constante `MODEL`
en `main.py` por `"text-embedding-ada-002"`. El modelo de este proyecto es
`text-embedding-3-small`, cuyo vector predeterminado tiene 1536 dimensiones.

## 5. Ejecutar las pruebas

La prueba usa un cliente falso: valida la llamada y no consume créditos ni
necesita una clave real.

```bash
uv run python -m unittest discover -s tests -v
```

## Referencias oficiales

- [Guía de embeddings de OpenAI](https://developers.openai.com/api/docs/guides/embeddings)
- [Referencia de `embeddings.create`](https://developers.openai.com/api/reference/resources/embeddings/methods/create)
- [SDK oficial de OpenAI para Python](https://developers.openai.com/api/reference/python)
- [Primer llamado a la API de DeepSeek](https://api-docs.deepseek.com/guides/codex)
- [Chat Completions de DeepSeek](https://api-docs.deepseek.com/api/create-chat-completion/)
