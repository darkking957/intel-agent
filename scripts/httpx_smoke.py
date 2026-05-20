import os

import httpx
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["DEEPSEEK_API_KEY"]
base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
model = os.getenv("LLM_MODEL", "deepseek-chat")

payload = {
    "model": model,
    "messages": [{"role": "user", "content": "用一句话介绍你自己。"}],
    "temperature": 0.2,
    "max_tokens": 128,
}

with httpx.Client(timeout=20.0) as client:
    resp = client.post(
        f"{base_url}/chat/completions",
        headers={"Authorization":f"Bearer {api_key}"},
        json=payload,
    )
    resp.raise_for_status()

data = resp.json()
print(data["choices"][0]["message"]["content"])