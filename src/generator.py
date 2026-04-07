import logging 
import os 
from pathlib import Path

import yaml

from src.models import RetrievedChunk , RagResponse

logger = logging.getLogger(__name__)

class ResponseGenerator:
    def __init__(
            self,
            llm_provider : str = "groq",
            model_name : str | None = None ,
            prompts_path : str | Path | None = None
    ):
        self.prompts = self._load_prompts(prompts_path)

        if llm_provider == 'groq':
            self._init_groq(model_name or 'llama-3.1-8b-instant')
        elif llm_provider == 'ollama':
            self._init_ollama(model_name or "llama3.1")
        else :
            raise ValueError(f"Unknown LLM Provider : {llm_provider}")
        
        self.llm_provider = llm_provider

    def _load_prompts(self,path: str | Path | None ) -> dict: 
        if path is None : 
            path = Path(__file__).parent.parent / "config" / "prompts.yaml"
        with open(path , "r") as f : 
            prompts = yaml.safe_load(f)
        logger.info(f"Loadded prompts v{prompts.get('version', 'unknown')}")
        return prompts

    def _init_groq(self,model_name: str) -> None:
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key :
            raise ValueError(
                "GROQ_API_KEY not set. Get your free key at https://console.groq.com"
            )
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        logger.info(f"LLM ready : Groq / {model_name}")
    
    def _init_ollama(self,model_name: str) -> None:
        import requests
        try :
            requests.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
        except Exception : 
            raise ConnectionError(
                "Ollama not running. Start with: ollama serve\n"
                f"Then: ollama pull {model_name}"
            )
        self.model_name = model_name
        logger.info(f"LLM ready: Ollama / {model_name}")

    def generate(self , query : str , retrieved_chunks : list[RetrievedChunk]) -> RagResponse :
        context = self._format_context(retrieved_chunks)
        system_prompt = self.prompts["system_prompt"]
        user_prompt = self.prompts["qa_prompt"].format(
            context=context,
            question=query,
        )

        if self.llm_provider == "groq":
            answer = self._call_groq(system_prompt , user_prompt)
        else : 
            answer = self._call_ollama(system_prompt,user_prompt)
        
        return RagResponse(
            answer = answer,
            retrieved_chunks = retrieved_chunks,
            query=query,
        )
    def _format_context(self , chunks:list[RetrievedChunk]) -> str:
        parts = []
        for i , chunk in enumerate(chunks,1):
            source = chunk.metadata.get("source","Unknown")
            page = chunk.metadata.get("page" , "")
            location = f" | Page {page}" if page else ""

            parts.append(
                f"[Source {i}] (File: {source}{location})\n"
                f"{chunk.content}"    
            )
        return "\n\n---\n\n".join(parts)
        
    def _call_groq(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
        model=self.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=1024,
        )
                
        return response.choices[0].message.content

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        import requests

        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
