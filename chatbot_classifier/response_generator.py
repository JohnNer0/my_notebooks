"""
Response Generator Module
Generates contextual responses based on intent and slots
"""

import random
from typing import Dict, List, Optional
from conversation_context import ConversationContext, Slot


class ResponseGenerator:
    """Generates bot responses based on intent and context"""
    
    def __init__(self):
        self.responses = self._load_response_templates()
    
    def _load_response_templates(self) -> Dict[str, List[str]]:
        """Load response templates for different intents"""
        return {
            'greeting': [
                "Hello! How can I help you today?",
                "Hi there! What can I do for you?",
                "Greetings! How may I assist you?",
                "Hey! What brings you here?",
            ],
            'goodbye': [
                "Goodbye! Have a great day!",
                "See you later! Thanks for chatting!",
                "Take care! Come back soon!",
                "Farewell! Have a wonderful day!",
            ],
            'help': [
                "I can help you with orders, customer support, product information, and more. What do you need?",
                "I'm here to assist! I can help with orders, support issues, or general information. What's your question?",
                "Sure! I can help with: placing orders, tracking shipments, technical support, or company info. What do you need?",
            ],
            'order': [
                "Great! I'd like to help you place an order. What product are you interested in?",
                "Perfect! Let's start your order. What would you like to buy?",
                "I'd be happy to help with your order. What items are you interested in?",
            ],
            'support': [
                "I'm sorry to hear you're experiencing an issue. Can you describe the problem in more detail?",
                "I apologize for the inconvenience. What seems to be the problem?",
                "Let's resolve this together. What's the issue you're facing?",
            ],
            'information': [
                "I'd be happy to provide information. What would you like to know about us?",
                "What specific information are you looking for?",
                "I can help with that. What would you like to know?",
            ],
            'fallback': [
                "I'm not quite sure I understand. Could you rephrase that?",
                "Sorry, I didn't catch that. Can you say it differently?",
                "I'm having trouble understanding. Could you clarify?",
                "Let me try to help - could you give me more details?",
            ],
            'fallback_escalate': [
                "I'm having trouble understanding your request. Let me connect you with a human agent who can better assist you.",
                "I apologize, but I'm not able to help with that right now. A customer service representative will be with you shortly.",
                "I understand this might be complex. Let me transfer you to someone who can provide better support.",
            ],
            'slot_request': {
                'name': "What's your name?",
                'email': "Could you provide your email address?",
                'product': "Which product would you like to order?",
                'quantity': "How many would you like?",
                'delivery_address': "Where should we deliver this?",
            },
            'confirmation': [
                "Just to confirm, {details}. Is that correct?",
                "So you're looking for {details}. Should I proceed?",
                "Let me make sure I got this right: {details}. Does that sound good?",
            ]
        }
    
    def generate_response(self, intent: str, context: ConversationContext, 
                         confidence: float, is_fallback: bool = False) -> str:
        """Generate a response based on intent and context"""
        
        # Handle low confidence with fallback
        if confidence < 0.3 or is_fallback:
            return self._generate_fallback_response(context)
        
        # Check for unfilled slots
        unfilled = context.get_unfilled_required_slots()
        if unfilled:
            return self._request_slot(unfilled[0])
        
        # Generate intent-based response
        if intent in self.responses:
            return random.choice(self.responses[intent])
        
        # Default fallback
        return random.choice(self.responses['fallback'])
    
    def _generate_fallback_response(self, context: ConversationContext) -> str:
        """Generate a fallback response when confidence is low"""
        # After multiple fallbacks, escalate
        if context.turn_count > 2 and context.get_context_flag('fallback_attempts', 0) > 2:
            context.set_context_flag('needs_escalation', True)
            return random.choice(self.responses['fallback_escalate'])
        
        fallback_attempts = context.get_context_flag('fallback_attempts', 0)
        context.set_context_flag('fallback_attempts', fallback_attempts + 1)
        
        return random.choice(self.responses['fallback'])
    
    def _request_slot(self, slot: Slot) -> str:
        """Generate a slot request message"""
        slot.request()
        
        if slot.exceeded_attempts():
            return f"I've asked about your {slot.name} several times. Would you like to continue or get help from an agent?"
        
        return self.responses['slot_request'].get(
            slot.name, 
            f"Could you provide your {slot.name}?"
        )
    
    def generate_confirmation(self, context: ConversationContext) -> str:
        """Generate a confirmation message with filled slots"""
        slot_details = []
        for slot in context.slots.values():
            if slot.status.value in ['filled', 'confirmed']:
                slot_details.append(f"{slot.name}: {slot.value}")
        
        details = ", ".join(slot_details)
        return random.choice(self.responses['confirmation']).format(details=details)
    
    def generate_followup(self, intent: str, context: ConversationContext) -> str:
        """Generate a follow-up question or suggestion"""
        followup_questions = {
            'order': "Would you like to know about our shipping times?",
            'support': "Is there anything else I can help you with?",
            'information': "Would you like to know more about our products?",
        }
        
        return followup_questions.get(intent, "Is there anything else I can help you with?")
