from typing import Dict, Any, Optional


class State:
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.is_terminal: bool = False
        self.transition = config.get("transition", {})

    def check_transition_ready(self, context: Dict[str, Any]) -> bool:
        """
        USER DEFINED STATES SHOULD OVERRRIDE THIS FUNCTION
        """
        raise NotImplementedError

    def run(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        USER DEFINED STATES SHOULD OVERRRIDE THIS FUNCTION
        """
        raise NotImplementedError
