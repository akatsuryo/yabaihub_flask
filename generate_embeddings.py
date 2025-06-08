import openai
import json
import os
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("functions.json", "r", encoding="utf-8") as f:
    functions = json.load(f)

for func in functions:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=func["description"]
    )
    embedding = response.data[0].embedding
    func["embedding"] = embedding

with open("functions_with_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(functions, f, ensure_ascii=False, indent=2)

print("✅ 埋め込みベクトルの作成が完了しました。")
