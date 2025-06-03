import gradio as gr
from typing import List, Dict

from core.research_agent import ResearchAgent


def format_sources(sources: List[Dict]) -> str:
    """Format sources for display with improved styling."""
    if not sources:
        return """
        <div style="padding: 24px; background: #f8fafc; border-radius: 12px; text-align: center; 
                   color: #64748b; font-family: 'Roboto', sans-serif; border: 1px solid #e2e8f0;">
            <div style="font-size: 20px; margin-bottom: 8px;">📚</div>
            <p style="margin: 0; font-size: 14px;">No sources were cited for this research.</p>
        </div>
        """

    formatted = """
    <div style="font-family: 'Roboto', sans-serif;">
        <h3 style="color: #1e293b; margin-bottom: 16px; font-weight: 600; font-size: 18px; 
                   display: flex; align-items: center;">
            <span style="margin-right: 8px;">📚</span>
            Sources & References
        </h3>
    """

    for i, source in enumerate(sources, 1):
        formatted += f"""
        <div style="background: white; border-radius: 8px; padding: 16px; margin-bottom: 12px; 
                   border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);">
            <div style="display: flex; align-items: flex-start; margin-bottom: 8px;">
                <span style="background: #3b82f6; color: white; border-radius: 50%; 
                           width: 20px; height: 20px; display: flex; align-items: center; 
                           justify-content: center; font-weight: 600; margin-right: 10px; 
                           font-size: 12px; flex-shrink: 0; margin-top: 2px;">{i}</span>
                <h4 style="margin: 0; color: #1e293b; font-weight: 600; font-size: 14px; line-height: 1.4;">
                    {source['title']}
                </h4>
            </div>
            <p style="color: #64748b; margin: 4px 0 8px 30px; font-size: 13px;">
                {source['source_name']}
            </p>
        """

        if source['url']:
            formatted += f"""
            <p style="margin: 8px 0 8px 30px;">
                <a href="{source['url']}" target="_blank" 
                   style="color: #3b82f6; text-decoration: none; font-size: 13px; font-weight: 500;">
                    🔗 View Source ↗
                </a>
            </p>
            """

        formatted += f"""
            <div style="background: #f8fafc; border-radius: 6px; padding: 12px; margin: 8px 0 0 30px;">
                <p style="margin: 0; color: #475569; line-height: 1.5; font-size: 13px;">
                    {source['snippet']}
                </p>
            </div>
        </div>
        """

    formatted += "</div>"
    return formatted


def create_custom_css() -> str:
    """Create custom CSS for the interface."""
    return """
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
        background: #f8fafc !important;
        min-height: 100vh;
        padding: 20px;
    }

    .header-section {
        background: white;
        border-radius: 12px;
        padding: 32px 24px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
    }

    .header-title {
        font-size: 2.25rem;
        font-weight: 700;
        margin-bottom: 8px;
        color: #1e293b;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .header-subtitle {
        font-size: 1rem;
        color: #64748b;
        font-weight: 400;
        line-height: 1.6;
        max-width: 600px;
        margin: 0 auto;
    }

    .query-section {
        background: white;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .examples-section {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .examples-title {
        color: #374151;
        font-weight: 600;
        margin-bottom: 12px;
        font-size: 16px;
    }

    .results-section {
        background: white;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .gr-textbox {
        border-radius: 8px !important;
        border: 2px solid #e2e8f0 !important;
        font-family: 'Roboto', sans-serif !important;
    }

    .gr-textbox:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }

    .gr-button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        text-transform: none !important;
    }

    .gr-button.primary {
        background: #3b82f6 !important;
        color: white !important;
        border: none !important;
    }

    .gr-button.primary:hover {
        background: #2563eb !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
    }

    .gr-button.secondary {
        background: white !important;
        color: #374151 !important;
        border: 2px solid #e2e8f0 !important;
    }

    .gr-button.secondary:hover {
        background: #f9fafb !important;
        border-color: #d1d5db !important;
    }

    .example-button {
        background: #f8fafc !important;
        color: #374151 !important;
        border: 1px solid #e2e8f0 !important;
        font-size: 13px !important;
        padding: 8px 12px !important;
        margin: 4px !important;
        border-radius: 6px !important;
    }

    .example-button:hover {
        background: #3b82f6 !important;
        color: white !important;
        border-color: #3b82f6 !important;
    }

    .status-success {
        background: #10b981;
        color: white;
        border-radius: 8px;
        padding: 12px 16px;
        font-weight: 500;
        font-size: 14px;
    }

    .status-error {
        background: #ef4444;
        color: white;
        border-radius: 8px;
        padding: 12px 16px;
        font-weight: 500;
        font-size: 14px;
    }

    .status-ready {
        background: #f8fafc;
        color: #64748b;
        border-radius: 8px;
        padding: 12px 16px;
        font-weight: 500;
        font-size: 14px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    """


def create_gradio_interface():
    """Create and configure the Gradio interface with modern design."""

    agent = ResearchAgent()

    def research_query(query: str, progress=gr.Progress()) -> tuple:
        """Handle research query from Gradio interface."""
        if not query or not query.strip():
            return (
                '<div class="status-error">⚠️ Please enter a valid research query.</div>',
                "## 📝 Ready to Research\n\nEnter your question above to get started!",
                '<div style="text-align: center; padding: 20px; color: #64748b; font-family: \'Roboto\', sans-serif;">💡 Sources will appear here after research</div>'
            )

        progress(0, desc="🚀 Initializing research...")

        def progress_callback(message: str):
            progress(0.5, desc=f"🔍 {message}")

        result = agent.process_query(query, progress_callback)

        progress(1.0, desc="✅ Research complete!")

        if result["success"]:
            summary = f"""
## 📜 Research Summary

**Query:** {result['query']}

---

{result['summary']}
            """
            sources = format_sources(result["sources"])
            status = f'<div class="status-success">✅ Research completed successfully</div>'
        else:
            summary = f"""
## ❌ Research Failed

{result.get('error', 'Unknown error occurred')}

Please try rephrasing your question or check your internet connection.
            """
            sources = '<div style="text-align: center; padding: 20px; color: #ef4444; font-family: \'Roboto\', sans-serif;">❌ No sources available due to research failure</div>'
            status = f'<div class="status-error">❌ Research failed: {result.get("error", "Unknown error")}</div>'

        return status, summary, sources

    def clear_interface():
        """Clear all interface elements."""
        return (
            "",
            '<div class="status-ready">🎯 Ready to research! Enter your question above.</div>',
            "## 📝 Ready to Research\n\nEnter your question above to get started with AI-powered research!",
            '<div style="text-align: center; padding: 32px; color: #94a3b8; font-family: \'Roboto\', sans-serif;"><div style="font-size: 32px; margin-bottom: 12px;">📚</div><p style="margin: 0;">Sources will appear here after research</p></div>'
        )

    # Create the interface with custom theme
    with gr.Blocks(
        title="🔍 AI Research Agent",
        css=create_custom_css(),
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Roboto")
        ),
        analytics_enabled=False
    ) as interface:

        # Header section
        gr.HTML("""
        <div class="header-section">
            <h1 class="header-title">🔍 AI Research Agent</h1>
            <p class="header-subtitle">
                Get comprehensive, well-sourced answers to your research questions using advanced AI that searches and analyzes information from multiple sources.
            </p>
        </div>
        """)

        # Query input section
        with gr.Group(elem_classes=["query-section"]):
            query_input = gr.Textbox(
                label="🎯 Your Research Question",
                placeholder="Ask me anything... e.g., 'What are the latest developments in renewable energy?'",
                lines=3,
                max_lines=5
            )

            with gr.Row():
                search_btn = gr.Button(
                    "🚀 Start Research",
                    variant="primary",
                    size="lg",
                    scale=3,
                    elem_classes=["primary"]
                )
                clear_btn = gr.Button(
                    "🗑️ Clear",
                    variant="secondary",
                    size="lg",
                    scale=1,
                    elem_classes=["secondary"]
                )

        # Examples section - moved closer to input
        with gr.Group(elem_classes=["examples-section"]):
            gr.HTML('<h3 class="examples-title">💡 Try These Example Queries</h3>')

            example_queries = [
                "What were the main causes of the French Revolution?",
                "Latest breakthroughs in quantum computing 2024",
                "How does climate change affect food security?",
                "AI applications in modern healthcare",
                "Economic impacts of renewable energy"
            ]

            with gr.Row():
                for example in example_queries[:3]:
                    gr.Button(
                        example,
                        size="sm",
                        elem_classes=["example-button"]
                    ).click(
                        lambda x=example: x,
                        outputs=query_input
                    )

            with gr.Row():
                for example in example_queries[3:]:
                    gr.Button(
                        example,
                        size="sm",
                        elem_classes=["example-button"]
                    ).click(
                        lambda x=example: x,
                        outputs=query_input
                    )

        # Status section
        status_output = gr.HTML(
            value='<div class="status-ready">🎯 Ready to research! Enter your question above.</div>'
        )

        # Results section
        with gr.Group(elem_classes=["results-section"]):
            with gr.Row():
                with gr.Column(scale=1):
                    summary_output = gr.Markdown(
                        value="## 📝 Ready to Research\n\nEnter your question above to get started with AI-powered research!",
                        label="Research Summary"
                    )

                with gr.Column(scale=1):
                    sources_output = gr.HTML(
                        value='<div style="text-align: center; padding: 32px; color: #94a3b8; font-family: \'Roboto\', sans-serif;"><div style="font-size: 32px; margin-bottom: 12px;">📚</div><p style="margin: 0;">Sources will appear here after research</p></div>',
                        label="Sources & References"
                    )

        # Event handlers
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

        clear_btn.click(
            fn=clear_interface,
            outputs=[query_input, status_output, summary_output, sources_output]
        )

    return interface
