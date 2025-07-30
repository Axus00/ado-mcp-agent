from langgraph.graph import END, StateGraph
from core.models.agent_state import AgentState
from config.personality import AI_NAME
from langchain_openai import ChatOpenAI
from services.tools.llm_selector_tool import LLMSelectorTool
import os
from dotenv import load_dotenv


load_dotenv()


llm_orquester = LLMSelectorTool()


llm = ChatOpenAI(
    model="gpt-4o", 
    api_key= os.getenv("OPENAI_API_KEY"), # type: ignore
    temperature=0.7
)

def select_llm(state: AgentState) -> AgentState: # type: ignore
    """Este nodo carga el asistente virtual"""
    llm_selected = state["llm_choice"]

    if llm_selected in ["openai", "mistralai"]:
        llms = llm_orquester._initialize_llms()
        state["llm_selected"] = llms.get(llm_selected)

        if not state["llm_selected"]:
            state["message_error"] = f"LLM '{llm_selected}' no disponible"
    else:
        state["llm_selected"] = None
        state["message_error"] = f"LLM '{llm_selected}' no reconocido"


    return state


def initialize_llm(state: AgentState) -> AgentState: # type: ignore
    user_message = state["user_input"]
    selected_llm = state["llm_selected"]

    if not selected_llm:
        state["response"] = "No se ha seleccionado un LLM válido"
        return state

    prompt = f"""

        Analiza de acuerdo al prompt proporcionado la
        solicitud del usuario

        Contenta de manera coherente, teniendo encuenta
        la siguiente personalidad: {AI_NAME}

        El usuario te ha preguntado: {user_message}
    """
    try:
        response = selected_llm(prompt) # type: ignore
        state["response"] = response # type: ignore
        return state
    except Exception as e:
        state["response"] = f"Error al procesar la solicitud: {str(e)}"

    return state

graph = StateGraph(AgentState)

graph.add_node("select_llm", select_llm)
graph.add_node("initialize_llm", initialize_llm)

graph.set_entry_point("select_llm")

graph.add_edge("select_llm", "initialize_llm")
graph.add_edge("initialize_llm", END)