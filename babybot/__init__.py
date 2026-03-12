from .config import Config
from .orchestrator import OrchestratorAgent
from .resource import ResourceManager
from .scheduler import Scheduler

__version__ = "0.1.0"
__all__ = ["Config", "OrchestratorAgent", "ResourceManager", "Scheduler", "main"]


def main():
    """Entry point for the babybot CLI."""
    from .cli import run

    run()
