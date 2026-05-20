from intel_agent.llm import LLMClient

client = LLMClient()

messages = [
    {"role": "system", "content": "你是一个严谨的 Python 工程导师。"},
    {"role": "user", "content": "用三句话解释什么是 OpenAI-compatible API。"},
]

for token in client.stream_text(messages=messages):
    print(token,end="",flush=True)

print()