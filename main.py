import gradio as gr
import os
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Disable Gradio analytics
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_TELEMETRY_ENABLED"] = "False"
os.environ["GRADIO_SHARE"] = "False"

from graph_builder import search_agent_graph
from prompts import MAIN_SYSTEM_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchAgent:
    """Research agent wrapper for handling queries and streaming responses."""

    def __init__(self):
        self.max_iterations = 20

    def process_query(self, query: str, progress_callback=None) -> Dict[str, Any]:
        """Process a research query and return results."""
        if not query or not query.strip():
            return {
                "success": False,
                "error": "Please enter a valid query.",
                "summary": "",
                "sources": []
            }
        
        try:
            logger.info(f"Processing query: {query}")
            
            initial_messages = [
                {"role": "system", "content": MAIN_SYSTEM_PROMPT},
                {"role": "user", "content": query.strip()}
            ]
            
            initial_state = {
                "query": query.strip(),
                "messages": initial_messages,
                "final_answer": None,
                "max_iterations": self.max_iterations,
                "current_iteration": 0
            }
            
            final_event_data = None
            terminal_nodes = ["prepare_final_answer", "force_no_info_iterations", "force_no_info_no_tool"]
            
            for event_count, event in enumerate(search_agent_graph.stream(
                initial_state,
                {"recursion_limit": self.max_iterations * 2 + 5}
            )):
                if progress_callback:
                    progress_callback(f"Processing step {event_count + 1}...")
                
                current_node = list(event.keys())[0]
                
                if current_node in terminal_nodes:
                    final_event_data = event[current_node]
                    if 'final_answer' in final_event_data and final_event_data['final_answer'] is not None:
                        break
            
            if final_event_data and final_event_data.get('final_answer'):
                final_summary = final_event_data['final_answer']
                sources_list = []
                
                if hasattr(final_summary, 'sources') and final_summary.sources:
                    for source in final_summary.sources:
                        source_info = {
                            "title": getattr(source, 'title', 'Unknown Title'),
                            "source_name": getattr(source, 'source_name', 'Unknown Source'),
                            "url": getattr(source, 'url', None),
                            "snippet": getattr(source, 'snippet', '')[:300] + "..." if getattr(source, 'snippet', '') else "No snippet available"
                        }
                        sources_list.append(source_info)
                
                return {
                    "success": True,
                    "summary": getattr(final_summary, 'summary', 'No summary available'),
                    "sources": sources_list,
                    "query": query.strip()
                }
            else:
                return {
                    "success": False,
                    "error": "No results found or agent failed to complete the research.",
                    "summary": "",
                    "sources": []
                }
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "success": False,
                "error": f"An error occurred: {str(e)}",
                "summary": "",
                "sources": []
            }

def create_gradio_interface():
    """Create and configure the Gradio interface."""
    
    agent = ResearchAgent()
    
    def format_sources(sources: List[Dict]) -> str:
        """Format sources for display."""
        if not sources:
            return "No sources were cited."
        
        formatted = "**Sources:**\n\n"
        for i, source in enumerate(sources, 1):
            formatted += f"**{i}. {source['title']}** ({source['source_name']})\n"
            if source['url']:
                formatted += f"   🔗 URL: {source['url']}\n"
            formatted += f"   📝 {source['snippet']}\n\n"

        return formatted

    def research_query(query: str, progress=gr.Progress()) -> tuple:
        """Handle research query from Gradio interface."""
        if not query or not query.strip():
            return "Please enter a valid research query.", "", ""
        
        progress(0, desc="Starting research...")
        
        def progress_callback(message: str):
            progress(0.5, desc=message)
        
        result = agent.process_query(query, progress_callback)
        
        progress(1.0, desc="Research complete!")
        
        if result["success"]:
            summary = f"## 📜 Research Summary\n\n{result['summary']}"
            sources = format_sources(result["sources"])
            status = f"✅ Research completed successfully for: \"{result['query']}\""
        else:
            summary = f"## ❌ Research Failed\n\n{result.get('error', 'Unknown error occurred')}"
            sources = ""
            status = f"❌ Research failed: {result.get('error', 'Unknown error')}"
        
        return status, summary, sources
    
    with gr.Blocks(
        title="AI Research Agent",
        theme=gr.themes.Soft(),
        analytics_enabled=False
    ) as interface:
        
        gr.Markdown(
            """
            # 🔍 AI Research Agent
            
            Enter your research query below and let the AI agent search and analyze information for you.
            The agent will use multiple sources including Wikipedia, web search, and other tools to provide comprehensive answers.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="Research Query",
                    placeholder="Enter your research question here...",
                    lines=3,
                    max_lines=5
                )
                
                search_btn = gr.Button(
                    "🚀 Start Research",
                    variant="primary",
                    size="lg"
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False
                )
        
        with gr.Row():
            with gr.Column():
                summary_output = gr.Markdown(
                    label="Research Summary",
                    value="Research results will appear here..."
                )
                
                sources_output = gr.Markdown(
                    label="Sources",
                    value="Sources will be listed here..."
                )
        
        gr.Markdown("### 💡 Example Queries")
        example_queries = [
            "What were the main causes of the French Revolution?",
            "What are the latest advancements in quantum computing in 2024?",
            "What is the capital of France and its current population?"
        ]
        
        for example in example_queries:
            gr.Button(example, size="sm").click(
                lambda x=example: x,
                outputs=query_input
            )
        
        search_btn.click(
            fn=research_query,
            inputs=[query_input],
            outputs=[status_output, summary_output, sources_output]
        )
        
        query_input.submit(
            fn=research_query,
            inputs=[query_input],
            outputs=[status_output, summary_output, sources_output]
        )
    
    return interface

def main():
    """Main function to launch the application."""
    host = os.getenv("GRADIO_HOST", "127.0.0.1")
    port = int(os.getenv("GRADIO_PORT", "7860"))
    
    interface = create_gradio_interface()
    
    logger.info(f"Starting Gradio interface on {host}:{port}")
    
    interface.launch(
        server_name=host,
        server_port=port,
        share=False,
        show_error=True,
        inbrowser=False
    )

if __name__ == "__main__":
    main()
