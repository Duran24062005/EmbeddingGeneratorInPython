from openai import OpenAI

client = OpenAI(
    api_key="apikey",
    base_url="https://api.ofox.ai/v1",
)

response = client.chat.completions.create(
    model="openai/gpt-5.4-mini",
    messages=[{"role": "user", "content": "What is the meaning of life?"}],
)
print(response.choices[0].message.content)