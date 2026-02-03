from typing import Dict, Any
from src.config import Config
import requests
import os


class AnswerGenerator:
    def __init__(self, model: str = None):
        self.model = model or os.getenv("GROQ_DEFAULT_MODEL", "llama-3.1-8b-instant")

    def _build_prompts(self, query: str, context: str) -> Dict[str, str]:
        system_prompt = (
            "You are a helpful assistant that answers questions based on provided context.\n"
            "Rules:\n"
            "- Answer based ONLY on the provided context\n"
            "- If the context doesn't contain enough information, say so clearly\n"
            "- Be concise and accurate\n"
            "- Cite sources when relevant\n"
        )

        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nPlease answer the question based on the context above."
        return {"system": system_prompt, "user": user_prompt}

    def generate(self, query: str, context: str) -> Dict[str, Any]:
        prompts = self._build_prompts(query, context)

        headers = {"Authorization": f"Bearer {Config.GROQ_API_KEY}", "Content-Type": "application/json"}
        url = f"{Config.GROQ_API_URL}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompts['system']},
                {"role": "user", "content": prompts['user']},
            ],
            "temperature": 0.7,
            "max_tokens": 500,
        }

        print(f"Generating answer using Groq model {self.model}...")
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        j = r.json()

        answer = None
        if isinstance(j, dict) and 'choices' in j:
            choices = j.get('choices', [])
            if choices and isinstance(choices[0], dict):
                message = choices[0].get('message', {})
                if isinstance(message, dict):
                    answer = message.get('content')

        if answer is None:
            answer = str(j)

        usage = j.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}) if isinstance(j, dict) else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        return {"query": query, "answer": answer, "model": self.model, "usage": usage}
