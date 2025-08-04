import time
from typing import Optional

class ClickManager:
    """
    Manages click detection with cooldown to prevent false triggers.
    """
    
    def __init__(self, cooldown_seconds: float = 0.3):
        """
        Initialize click manager.
        
        Args:
            cooldown_seconds: Minimum time between clicks
        """
        self.last_click_time: float = 0.0
        self.cooldown_seconds: float = cooldown_seconds
        self.is_clicking: bool = False
    
    def can_click(self) -> bool:
        """
        Check if enough time has passed since last click.
        
        Returns:
            True if click is allowed, False otherwise
        """
        current_time = time.time()
        return (current_time - self.last_click_time) > self.cooldown_seconds
    
    def register_click(self) -> None:
        """Register that a click has occurred."""
        self.last_click_time = time.time()
        self.is_clicking = True
    
    def reset_click_state(self) -> None:
        """Reset the clicking state."""
        self.is_clicking = False