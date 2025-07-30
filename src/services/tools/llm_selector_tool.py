from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from pydantic import PrivateAttr
from typing import Any, Dict
import os

load_dotenv()

class LLMSelectorTool(BaseTool):
    name: str = "llm_selector"
    description: str = "Selecciona y ejecuta el LLM"

    _llm_instances: Dict[str, Any] = PrivateAttr()

    def __init__(self):
        super().__init__()
        self._llm_instances = self._initialize_llms()

    def _initialize_llms(self):
        llms = {}

        if os.getenv("OPENAI_API_KEY"):
            llms["openai"] = ChatOpenAI(
                model="gpt-4o",
                api_key=os.getenv("OPENAI_API_KEY"),  # type: ignore
                temperature=0.7
            )

        if os.getenv("MISTRAL_API_KEY"):
            llms["mistralai"] = ChatMistralAI(
                model_name="open-mistral-nemo",
                api_key=os.getenv("MISTRAL_API_KEY"),  # type: ignore
                temperature=0.7
            )

        return llms

    def _run(self, llm_name: str) -> Any:
        if llm_name not in self._llm_instances:
            return f"Modelo '{llm_name}' no está disponible."

        llm = self._llm_instances[llm_name]
        response = llm("Hola, ¿puedes decirme tu nombre?")
        return response.content
