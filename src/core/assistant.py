from core.graph import graph
from core.models.agent_state import AgentState
from langgraph.checkpoint.memory import InMemorySaver


memory = InMemorySaver()

compiled_graph = graph.compile(checkpointer=memory)

def _assistant():
    print("Selecciona un asistente virtual: openai, mistralai")
    llm_choice = input("✅ LLM: ").strip().lower()


    prompt = input("👱 Usuario: ")

    initial_agent_state = AgentState(
        user_input = prompt, 
        llm_choice=llm_choice,
        response='') # type: ignore

    config = {"configurable": {"thread_id": "test"}}

    final_state = compiled_graph.invoke(initial_agent_state, config=config)

    if "error" in final_state:
        print("❌ Error:", final_state["error"])
    else:
        print("🤖 Asistente:", final_state["response"])