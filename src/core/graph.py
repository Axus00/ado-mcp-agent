from langgraph.graph import END, StateGraph
from models.agent_state import AgentState

from data.config.personality import DORIS_PERSONALITY


def user_prompt(state: AgentState) -> AgentState:
    """Este nodo recibe el prompt por parte del usuario"""

    state["user_input"] = state["user_input"]
    return state

def initialize_llm(state: AgentState) -> AgentState:
    """Este nodo carga el asistente virtual"""
    user_message = state["user_input"]

    state["response"] = f"""

        Analiza de acuerdo al prompt proporcionado la
        solicitud del usuario

        Contenta de manera coherente, teniendo encuenta
        la siguiente personalidad: {DORIS_PERSONALITY}

        El usuario te ha preguntado: {user_message}
    """
    return state

graph = StateGraph(AgentState)

graph.add_node("user_input", user_prompt)
graph.add_node("initialize_llm", initialize_llm)

graph.set_entry_point("user_input")

graph.add_edge("user_input", "initialize_llm")
graph.add_edge("initialize_llm", END)