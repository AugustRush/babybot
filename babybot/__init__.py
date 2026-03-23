from .config import Config
from .orchestrator import OrchestratorAgent
from .resource import ResourceManager

__version__ = "0.1.0"
__all__ = ["Config", "OrchestratorAgent", "ResourceManager", "main", "gateway"]


def main():
    """Entry point for the interactive babybot CLI."""
    from .cli import run

    run()


def gateway():
    """Entry point for the channel gateway (starts all enabled channels)."""
    from .cli import run_gateway

    run_gateway()
