from .config import Config
from .orchestrator import OrchestratorAgent
from .resource import ResourceManager

__version__ = "0.1.0"
__all__ = ["Config", "OrchestratorAgent", "ResourceManager", "main"]


def main():
    """Entry point for the babybot CLI."""
    from .cli import run

    run()
