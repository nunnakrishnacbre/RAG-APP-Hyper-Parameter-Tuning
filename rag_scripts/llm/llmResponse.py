import logs
import traceback
from typing import List, Dict, Any, Optional
from groq import Groq

from rag_scripts.interfaces import ILLM
from configuration import Configuration

class GROQLLM(ILLM):
    def __init__(self, api_key: str = Configuration.GROQ_API_KEY,
                 model_name: str = Configuration.DEFAULT_GROQ_LLM_MODEL):
        if not api_key:
            raise ValueError("Groq API not provided in env file")
        
        self.api_key = api_key
        self.model_name = model_name
        self.client = Groq(api_key=self.api_key)
        self.default_system_message = ("You are a helpful assistant for Flykite Airline HT Policy document queries."
                                       "provide a concise and accurate answers based strictly on the provided context."
                                       "Do not hallucinate or add ungrounded details.")

        logs.logger.info(f"Initialized Groq LLM with model: {self.model_name}")

    def generate_response(self, prompt, system_message: Optional[str] = None,
                          context: Optional[List[Dict]] = None,
                          temperature: float = 0.1,
                          top_p: float = 0.95,
                          max_tokens: int = 1000) -> str:
        try:
            complete_prompt = prompt
            if context:
                context_text ="\n".join([doc['content'] for doc in context])
                complete_prompt = f"Context:\n{context_text}\n\nQuestion: {prompt}"
            
            message = [{"role": "system", "content":system_message or self.default_system_message},
                       {"role":"user","content":complete_prompt} ]
            
            completion = self.client.chat.completions.create(
                model = self.model_name if self.model_name else Configuration.DEFAULT_GROQ_LLM_MODEL,
                messages= message,
                temperature=temperature if temperature is not None else 0.1,
                max_tokens=max_tokens if max_tokens is not None else 1000,
                top_p=top_p if top_p is not None else 0.95,
                stream=True, stop=None
            )

            response_content = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    response_content+=chunk.choices[0].delta.content
            return response_content.strip()

        except Exception as ex:
            logs.logger.error(f"Exception in LLM response: {ex}")
            logs.logger.error(traceback.print_exc())
