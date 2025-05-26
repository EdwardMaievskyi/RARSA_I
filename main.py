from graph_builder import search_agent_graph
from prompts import MAIN_SYSTEM_PROMPT


queries = [
    "What were the main causes of the French Revolution?", # Good for Wikipedia
    "What are the latest advancements in quantum computing in 2024?", # Good for DuckDuckGo/Tavily
    "Summarize the main points from the Wikipedia page on 'Photosynthesis'.", # Tests Wikipedia then implies scraping (though Wikipedia tool is direct)
    "Find reviews for the latest iPhone model.", # DuckDuckGo / Tavily
    "What is the capital of France and its current population?", # Wikipedia
    "Is there any information on the fictional Flumph creature from D&D on the web?", # Might find some, might not
    "asdfqweriuqwehriasdfkljqwer", # Likely no results
    "What are the lates news related to Russo-Ukrainian war?" # Good for DuckDuckGo/Tavily
    ]

query_to_run = queries[7] # Select a query to run
print(f"🚀 Initializing research for query: \"{query_to_run}\"")

    # Initial state

initial_messages = [
    {"role": "system", "content": MAIN_SYSTEM_PROMPT},
    {"role": "user", "content": query_to_run}
    ]

initial_state = {
    "query": query_to_run,
    "messages": initial_messages,
    "final_answer": None,
    "max_iterations": 20,
    "current_iteration": 0
    }

print("\n--- Running Research Agent ---")
final_event = None
final_event_data = None  # Initialize final_event_data outside the loop

for event_count, event in enumerate(search_agent_graph.stream(initial_state,
                                                              {"recursion_limit": initial_state["max_iterations"] * 2 + 5})):
        # print(f"\n--- Event {event_count + 1} ---")
        # print(event) # Print each event in the stream
        # The final event will contain the 'final_answer'
        # The key in the event dictionary corresponds to the node name that just executed.

    # Check if the current event corresponds to one of the terminal nodes
    terminal_nodes = ["prepare_final_answer", "force_no_info_iterations", "force_no_info_no_tool"]
    current_node = list(event.keys())[0] # Get the name of the node that just ran

    if current_node in terminal_nodes:
        final_event_data = event[current_node]
        # We expect the final_answer to be populated by these nodes.
        # Break the loop once we hit a terminal node and its data contains 'final_answer'.
        if 'final_answer' in final_event_data and final_event_data['final_answer'] is not None:
            final_event = final_event_data  # Capture the state when final_answer is populated
            break  # Stop streaming once we have the final answer


print("\n\n--- Research Complete ---")

# Access final_event_data outside the loop; it's now guaranteed to be initialized
if final_event_data and final_event_data.get('final_answer'):
  final_summary: ResearchSummary = final_event_data['final_answer'] # Use final_event_data here
  print("\n## 📜 Final Research Summary:")
  print(f"**Summary:** {final_summary.summary}")
  if final_summary.sources:
    print("\n**Sources:**")
    for i, source in enumerate(final_summary.sources):
      print(f"  {i+1}. **{source.title}** ({source.source_name})")
      print(f"     URL: {source.url if source.url else 'N/A'}")
      print(f"     Snippet: {source.snippet[:200]}..." if source.snippet else "N/A") # Print a shorter snippet
  else:
    print("\n**Sources:** No sources were cited.")
else:
  print("No final answer was produced by the agent, or streaming ended before final answer.")
  # If final_event_data was initialized but didn't have 'final_answer', print its content for debugging
  print("Last known event data:", final_event_data if final_event_data else "No terminal event data captured.")
