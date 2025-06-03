import os
import logging
from dotenv import load_dotenv

# Load environment variables should be done before other imports
load_dotenv()

from web_ui.gradio_interface import create_gradio_interface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
