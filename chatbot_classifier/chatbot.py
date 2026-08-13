"""
Main Chatbot Module
Orchestrates intent classification, conversation logic, and response generation
"""

import os
from typing import Dict, Optional, Tuple
from intent_classifier import IntentClassifier
from conversation_context import ConversationContext
from response_generator import ResponseGenerator


class Chatbot:
    """Main chatbot class that orchestrates all components"""
    
    def __init__(self, model_path: str = 'models/intent_model.pkl'):
        self.classifier = IntentClassifier(model_path)
        self.response_generator = ResponseGenerator()
        self.conversations: Dict[str, ConversationContext] = {}
        self.model_path = model_path
        self.max_fallback_attempts = 3
        
        # Try to load existing model, otherwise create new one
        if os.path.exists(model_path):
            self.classifier.load()
        else:
            self.classifier.train()
            self.classifier.save()
    
    def start_conversation(self, user_id: str) -> ConversationContext:
        """Start a new conversation with a user"""
        context = ConversationContext(user_id)
        
        # Define slots based on common order flow
        context.add_slot('name', required=True)
        context.add_slot('product', required=True)
        context.add_slot('quantity', required=True)
        context.add_slot('email', required=False)
        
        self.conversations[user_id] = context
        return context
    
    def get_conversation(self, user_id: str) -> Optional[ConversationContext]:
        """Get an existing conversation"""
        if user_id not in self.conversations:
            self.start_conversation(user_id)
        return self.conversations[user_id]
    
    def process_message(self, user_id: str, user_message: str, 
                       confidence_threshold: float = 0.3) -> Tuple[str, ConversationContext]:
        """Process user message and generate response"""
        
        # Get or create conversation context
        context = self.get_conversation(user_id)
        
        # Classify intent
        prediction = self.classifier.predict(user_message, confidence_threshold)
        intent = prediction['intent']
        confidence = prediction['confidence']
        is_below_threshold = prediction['below_threshold']
        
        # Update context
        context.current_intent = intent
        
        # Handle low confidence with fallback
        if is_below_threshold:
            response = self._handle_low_confidence(context, user_message)
        else:
            # Extract entities and fill slots if possible
            self._extract_and_fill_slots(context, user_message, intent)
            
            # Generate response
            response = self.response_generator.generate_response(
                intent, context, confidence, is_fallback=False
            )
            
            # Reset fallback counter on successful classification
            context.set_context_flag('fallback_attempts', 0)
        
        # Handle conversation logic based on intent
        response = self._handle_intent_logic(context, intent, response)
        
        # Add to history
        context.add_message(user_message, response)
        
        return response, context
    
    def _extract_and_fill_slots(self, context: ConversationContext, 
                               user_message: str, intent: str):
        """Extract information from user message and fill slots"""
        message_lower = user_message.lower()
        
        # Simple entity extraction (in production, use NER)
        if intent == 'order':
            # Look for common product names
            products = ['laptop', 'phone', 'tablet', 'headphones', 'monitor']
            for product in products:
                if product in message_lower:
                    context.fill_slot('product', product)
                    break
            
            # Look for quantities
            if 'one' in message_lower:
                context.fill_slot('quantity', 1)
            elif 'two' in message_lower:
                context.fill_slot('quantity', 2)
        
        # Extract name
        if 'my name is' in message_lower:
            parts = message_lower.split('my name is')
            if len(parts) > 1:
                name = parts[1].strip().split()[0].capitalize()
                context.fill_slot('name', name)
    
    def _handle_low_confidence(self, context: ConversationContext, 
                              user_message: str) -> str:
        """Handle messages with low confidence"""
        fallback_attempts = context.get_context_flag('fallback_attempts', 0)
        
        if fallback_attempts >= self.max_fallback_attempts:
            context.set_context_flag('needs_escalation', True)
            return "I'm having trouble understanding. Let me connect you with a human agent."
        
        context.set_context_flag('fallback_attempts', fallback_attempts + 1)
        return self.response_generator._generate_fallback_response(context)
    
    def _handle_intent_logic(self, context: ConversationContext, 
                            intent: str, base_response: str) -> str:
        """Apply conversation logic based on intent"""
        
        if intent == 'greeting':
            # Start fresh after greeting
            context.set_context_flag('greeted', True)
        
        elif intent == 'order':
            # Check if we have all required info
            if context.all_slots_filled():
                return self.response_generator.generate_confirmation(context)
            # Request next pending slot
            unfilled = context.get_unfilled_required_slots()
            if unfilled:
                return self.response_generator._request_slot(unfilled[0])
        
        elif intent == 'support':
            # Set flag for support interaction
            context.set_context_flag('in_support_flow', True)
        
        elif intent == 'goodbye':
            # Prepare for conversation end
            context.set_context_flag('conversation_ended', True)
        
        return base_response
    
    def get_pending_slots(self, user_id: str) -> list:
        """Get pending slots for a user"""
        context = self.get_conversation(user_id)
        return [s.name for s in context.get_unfilled_required_slots()]
    
    def get_conversation_summary(self, user_id: str) -> Dict:
        """Get conversation summary"""
        context = self.get_conversation(user_id)
        return context.get_summary()
    
    def end_conversation(self, user_id: str):
        """End a conversation"""
        if user_id in self.conversations:
            del self.conversations[user_id]
    
    def train_model(self):
        """Retrain the classifier model"""
        self.classifier.train()
        self.classifier.save()
    
    def evaluate_model(self):
        """Evaluate model performance"""
        self.classifier.evaluate()
