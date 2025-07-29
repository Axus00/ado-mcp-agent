from core.graph import graph
from core.models.agent_state import AgentState
from langgraph.checkpoint.memory import InMemorySaver


memory = InMemorySaver()

compiled_graph = graph.compile(checkpointer=memory)

def _assistant():
    if __name__ == "__main__":
        prompt = input("👱 Usuario: ")

        initial_agent_state = AgentState(user_input = prompt)

        final_state = compiled_graph.invoke(initial_agent_state)

        print("🤖 Asistente: ", final_state["response"])