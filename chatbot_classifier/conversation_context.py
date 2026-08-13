"""
Conversation Context Module
Manages conversation state, slots, and context
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum


class SlotStatus(Enum):
    """Status of a conversation slot"""
    EMPTY = "empty"
    PENDING = "pending"
    FILLED = "filled"
    CONFIRMED = "confirmed"


class Slot:
    """Represents a conversation slot (e.g., user name, order items)"""
    
    def __init__(self, name: str, required: bool = True, slot_type: str = "str"):
        self.name = name
        self.required = required
        self.slot_type = slot_type
        self.value = None
        self.status = SlotStatus.EMPTY
        self.attempts = 0
        self.max_attempts = 3
    
    def set_value(self, value: Any):
        """Set slot value and mark as filled"""
        self.value = value
        self.status = SlotStatus.FILLED
        self.attempts = 0
    
    def request(self):
        """Increment attempt counter when requesting the slot"""
        self.status = SlotStatus.PENDING
        self.attempts += 1
    
    def is_valid(self) -> bool:
        """Check if slot is valid (filled if required, or not required)"""
        if self.required:
            return self.status in [SlotStatus.FILLED, SlotStatus.CONFIRMED]
        return True
    
    def exceeded_attempts(self) -> bool:
        """Check if maximum attempts exceeded"""
        return self.attempts > self.max_attempts
    
    def __repr__(self):
        return f"Slot({self.name}={self.value}, status={self.status.value})"


class ConversationContext:
    """Manages conversation state and context"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.session_start = datetime.now()
        self.conversation_history: List[Dict[str, str]] = []
        self.current_intent = None
        self.slots: Dict[str, Slot] = {}
        self.followup_intent = None
        self.context_flags: Dict[str, bool] = {}
        self.turn_count = 0
    
    def add_slot(self, name: str, required: bool = True, slot_type: str = "str"):
        """Add a slot to the context"""
        self.slots[name] = Slot(name, required, slot_type)
    
    def get_slot(self, name: str) -> Optional[Slot]:
        """Get a slot by name"""
        return self.slots.get(name)
    
    def fill_slot(self, name: str, value: Any):
        """Fill a slot with a value"""
        if name in self.slots:
            self.slots[name].set_value(value)
    
    def get_pending_slots(self) -> List[Slot]:
        """Get all slots that are pending"""
        return [s for s in self.slots.values() if s.status == SlotStatus.PENDING]
    
    def get_unfilled_required_slots(self) -> List[Slot]:
        """Get all required slots that are not filled"""
        return [s for s in self.slots.values() 
                if s.required and s.status in [SlotStatus.EMPTY, SlotStatus.PENDING]]
    
    def all_slots_filled(self) -> bool:
        """Check if all required slots are filled"""
        return all(s.is_valid() for s in self.slots.values())
    
    def add_message(self, user_message: str, bot_response: str):
        """Add a message exchange to conversation history"""
        self.conversation_history.append({
            'user': user_message,
            'bot': bot_response,
            'timestamp': datetime.now().isoformat()
        })
        self.turn_count += 1
    
    def set_context_flag(self, flag_name: str, value: bool):
        """Set a context flag (e.g., 'order_confirmed')"""
        self.context_flags[flag_name] = value
    
    def get_context_flag(self, flag_name: str, default: bool = False) -> bool:
        """Get a context flag"""
        return self.context_flags.get(flag_name, default)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation state"""
        return {
            'user_id': self.user_id,
            'turn_count': self.turn_count,
            'current_intent': self.current_intent,
            'slots': {name: {'value': s.value, 'status': s.status.value} 
                     for name, s in self.slots.items()},
            'context_flags': self.context_flags,
            'unfilled_slots': [s.name for s in self.get_unfilled_required_slots()]
        }
    
    def reset(self):
        """Reset conversation context"""
        self.conversation_history = []
        self.current_intent = None
        self.slots = {}
        self.followup_intent = None
        self.context_flags = {}
        self.turn_count = 0
