from typing import TypedDict, Sequence, Dict, Any, Optional
from langchain_openai import ChatOpenAI

class AgentState(TypedDict):
    user_input: Dict[str, Any]
    response: str | Sequence[str] | ChatOpenAI
    llm_selected: Optional[str]
    llm_choice: Optional[str]
    message_error: Optional[str]
