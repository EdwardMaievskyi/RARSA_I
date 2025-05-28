from langgraph.graph import StateGraph, START, END
from core.nodes import call_model, call_tool, prepare_final_answer_node, \
    force_no_info_finish_node, should_continue
from core.state_models import AgentState


workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)
workflow.add_node("prepare_final_answer", prepare_final_answer_node)
workflow.add_node("force_no_info_iterations",
                  lambda s: force_no_info_finish_node(s, reason="Maximum iterations reached."))
workflow.add_node("force_no_info_no_tool",
                  lambda s: force_no_info_finish_node(s, reason="LLM provided a direct response or error instead of calling ResearchSummary."))


# Set the entrypoint
workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue_with_tool": "action",
        "prepare_final_answer": "prepare_final_answer",
        "force_finish_due_to_iterations": "force_no_info_iterations",
        "force_finish_no_tool_call": "force_no_info_no_tool",
    },
)

# Add edges for normal flow and end paths
workflow.add_edge("action", "agent") # Loop back after action
workflow.add_edge("prepare_final_answer", END)
workflow.add_edge("force_no_info_iterations", END)
workflow.add_edge("force_no_info_no_tool", END)


# Compile the graph
search_agent_graph = workflow.compile()
