
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    """Abstract base class for all specialized agents."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Processes the input and returns a structured output."""
        pass

    def get_info(self) -> Dict[str, str]:
        """Returns basic information about the agent."""
        return {"name": self.name, "description": self.description}

    # Common utilities like logging, metrics can be added here
    # For example:
    # def _log_request(self, request_id: str, payload: Dict[str, Any]):
    #     print(f"[{self.name}] Processing request {request_id} with payload: {payload}")

    # def _log_response(self, request_id: str, response: Dict[str, Any]):
    #     print(f"[{self.name}] Request {request_id} completed with response: {response}")

