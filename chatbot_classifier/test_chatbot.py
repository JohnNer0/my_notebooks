"""
Unit tests for the chatbot components
Run with: python -m pytest test_chatbot.py -v
"""

import pytest
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from intent_classifier import IntentClassifier
from conversation_context import ConversationContext, Slot, SlotStatus
from response_generator import ResponseGenerator
from chatbot import Chatbot


class TestIntentClassifier:
    """Test Intent Classifier"""
    
    @pytest.fixture
    def classifier(self):
        classifier = IntentClassifier()
        classifier.train()
        return classifier
    
    def test_classifier_trains(self, classifier):
        """Test that classifier trains successfully"""
        assert classifier.model is not None
        assert classifier.intents is not None
        assert len(classifier.intents) > 0
    
    def test_greeting_classification(self, classifier):
        """Test greeting intent classification"""
        result = classifier.predict("hello")
        assert result['intent'] == 'greeting'
        assert result['confidence'] > 0.3
    
    def test_order_classification(self, classifier):
        """Test order intent classification"""
        result = classifier.predict("i want to buy a laptop")
        assert result['intent'] == 'order'
    
    def test_support_classification(self, classifier):
        """Test support intent classification"""
        result = classifier.predict("my order is late")
        assert result['intent'] == 'support'
    
    def test_low_confidence_detection(self, classifier):
        """Test low confidence detection"""
        result = classifier.predict("xyzabc123")
        assert result['below_threshold'] is True or result['confidence'] < 0.5
    
    def test_model_save_load(self, tmp_path):
        """Test model serialization"""
        model_path = tmp_path / "test_model.pkl"
        
        # Train and save
        classifier = IntentClassifier(str(model_path))
        classifier.train()
        classifier.save()
        
        # Load and test
        classifier2 = IntentClassifier(str(model_path))
        classifier2.load()
        
        assert classifier2.model is not None
        assert classifier2.intents is not None


class TestConversationContext:
    """Test Conversation Context"""
    
    @pytest.fixture
    def context(self):
        return ConversationContext("test_user")
    
    def test_context_creation(self, context):
        """Test context is created properly"""
        assert context.user_id == "test_user"
        assert context.turn_count == 0
        assert len(context.conversation_history) == 0
    
    def test_add_slot(self, context):
        """Test adding slots"""
        context.add_slot('name', required=True)
        assert 'name' in context.slots
        assert context.slots['name'].required is True
    
    def test_slot_filling(self, context):
        """Test filling slots"""
        context.add_slot('name', required=True)
        context.fill_slot('name', 'John')
        
        slot = context.get_slot('name')
        assert slot.value == 'John'
        assert slot.status == SlotStatus.FILLED
    
    def test_required_slots_validation(self, context):
        """Test required slots validation"""
        context.add_slot('name', required=True)
        context.add_slot('email', required=False)
        
        assert not context.all_slots_filled()
        
        context.fill_slot('name', 'John')
        assert context.all_slots_filled()  # email is optional
    
    def test_pending_slots(self, context):
        """Test pending slots tracking"""
        context.add_slot('name', required=True)
        context.add_slot('email', required=True)
        
        slot = context.get_slot('name')
        slot.request()
        
        pending = context.get_pending_slots()
        assert len(pending) == 1
        assert pending[0].name == 'name'
    
    def test_unfilled_required_slots(self, context):
        """Test getting unfilled required slots"""
        context.add_slot('name', required=True)
        context.add_slot('email', required=False)
        context.add_slot('phone', required=True)
        
        context.fill_slot('name', 'John')
        
        unfilled = context.get_unfilled_required_slots()
        assert len(unfilled) == 1
        assert unfilled[0].name == 'phone'
    
    def test_conversation_history(self, context):
        """Test conversation history tracking"""
        context.add_message("hello", "hi there")
        context.add_message("how are you", "i'm good")
        
        assert len(context.conversation_history) == 2
        assert context.turn_count == 2
        assert context.conversation_history[0]['user'] == "hello"
    
    def test_context_flags(self, context):
        """Test context flags"""
        context.set_context_flag('greeting_done', True)
        assert context.get_context_flag('greeting_done') is True
        assert context.get_context_flag('nonexistent', False) is False
    
    def test_context_summary(self, context):
        """Test context summary generation"""
        context.add_slot('name', required=True)
        context.fill_slot('name', 'John')
        context.current_intent = 'order'
        
        summary = context.get_summary()
        assert summary['user_id'] == 'test_user'
        assert summary['current_intent'] == 'order'
        assert 'slots' in summary


class TestSlot:
    """Test Slot functionality"""
    
    def test_slot_creation(self):
        """Test slot creation"""
        slot = Slot('name', required=True)
        assert slot.name == 'name'
        assert slot.status == SlotStatus.EMPTY
        assert slot.value is None
    
    def test_slot_value_setting(self):
        """Test setting slot value"""
        slot = Slot('name')
        slot.set_value('John')
        
        assert slot.value == 'John'
        assert slot.status == SlotStatus.FILLED
    
    def test_slot_request_attempts(self):
        """Test slot request attempt tracking"""
        slot = Slot('name')
        
        for i in range(3):
            slot.request()
            assert slot.status == SlotStatus.PENDING
            assert slot.attempts == i + 1
        
        assert not slot.exceeded_attempts()
        
        slot.request()
        assert slot.exceeded_attempts()
    
    def test_slot_validation(self):
        """Test slot validation"""
        required_slot = Slot('name', required=True)
        assert not required_slot.is_valid()
        
        required_slot.set_value('John')
        assert required_slot.is_valid()
        
        optional_slot = Slot('email', required=False)
        assert optional_slot.is_valid()


class TestResponseGenerator:
    """Test Response Generator"""
    
    @pytest.fixture
    def generator(self):
        return ResponseGenerator()
    
    @pytest.fixture
    def context(self):
        return ConversationContext("test_user")
    
    def test_generator_initialization(self, generator):
        """Test generator initializes with responses"""
        assert generator.responses is not None
        assert 'greeting' in generator.responses
        assert 'fallback' in generator.responses
    
    def test_generate_response_greeting(self, generator, context):
        """Test generating greeting response"""
        response = generator.generate_response('greeting', context, confidence=0.9)
        assert response is not None
        assert len(response) > 0
    
    def test_generate_response_order(self, generator, context):
        """Test generating order response"""
        response = generator.generate_response('order', context, confidence=0.9)
        assert response is not None
        assert len(response) > 0
    
    def test_fallback_response(self, generator, context):
        """Test fallback response generation"""
        response = generator._generate_fallback_response(context)
        assert response is not None
        assert 'understand' in response.lower() or 'clarify' in response.lower()
    
    def test_slot_request(self, generator, context):
        """Test slot request generation"""
        context.add_slot('name', required=True)
        slot = context.get_slot('name')
        
        response = generator._request_slot(slot)
        assert response is not None
        assert slot.status == SlotStatus.PENDING
        assert slot.attempts == 1
    
    def test_generate_confirmation(self, generator, context):
        """Test confirmation message generation"""
        context.add_slot('name', required=True)
        context.add_slot('product', required=True)
        context.fill_slot('name', 'John')
        context.fill_slot('product', 'laptop')
        
        response = generator.generate_confirmation(context)
        assert 'John' in response or 'name' in response.lower()
        assert 'laptop' in response or 'product' in response.lower()
    
    def test_followup_generation(self, generator, context):
        """Test follow-up question generation"""
        response = generator.generate_followup('order', context)
        assert response is not None
        assert len(response) > 0


class TestChatbot:
    """Test Main Chatbot"""
    
    @pytest.fixture
    def chatbot(self, tmp_path):
        model_path = tmp_path / "test_model.pkl"
        chatbot = Chatbot(str(model_path))
        return chatbot
    
    def test_chatbot_initialization(self, chatbot):
        """Test chatbot initializes properly"""
        assert chatbot.classifier is not None
        assert chatbot.response_generator is not None
        assert len(chatbot.conversations) == 0
    
    def test_start_conversation(self, chatbot):
        """Test starting a conversation"""
        context = chatbot.start_conversation("user_001")
        assert context is not None
        assert context.user_id == "user_001"
        assert 'name' in context.slots
    
    def test_get_conversation(self, chatbot):
        """Test getting conversation"""
        chatbot.start_conversation("user_001")
        context = chatbot.get_conversation("user_001")
        assert context is not None
        assert context.user_id == "user_001"
    
    def test_process_greeting_message(self, chatbot):
        """Test processing a greeting message"""
        response, context = chatbot.process_message("user_001", "hello")
        assert response is not None
        assert len(response) > 0
        assert context.turn_count == 1
    
    def test_process_order_message(self, chatbot):
        """Test processing an order message"""
        response, context = chatbot.process_message("user_001", "i want to place an order")
        assert response is not None
        assert context.current_intent == 'order'
    
    def test_conversation_flow(self, chatbot):
        """Test multi-turn conversation flow"""
        messages = [
            "hello",
            "i want to buy something",
            "my name is John",
        ]
        
        for msg in messages:
            response, context = chatbot.process_message("user_001", msg)
            assert response is not None
            assert len(response) > 0
        
        assert context.turn_count == len(messages)
    
    def test_slot_filling_in_conversation(self, chatbot):
        """Test slot filling during conversation"""
        chatbot.process_message("user_001", "I want to order a laptop")
        chatbot.process_message("user_001", "My name is Alice")
        
        context = chatbot.get_conversation("user_001")
        name_slot = context.get_slot('name')
        
        # Should have extracted or requested the name
        assert name_slot is not None
    
    def test_pending_slots(self, chatbot):
        """Test getting pending slots"""
        chatbot.process_message("user_001", "hello")
        
        pending = chatbot.get_pending_slots("user_001")
        # Should return list (even if empty)
        assert isinstance(pending, list)
    
    def test_conversation_summary(self, chatbot):
        """Test getting conversation summary"""
        chatbot.process_message("user_001", "hello")
        chatbot.process_message("user_001", "i want to order")
        
        summary = chatbot.get_conversation_summary("user_001")
        assert summary is not None
        assert 'turn_count' in summary
        assert summary['turn_count'] == 2
    
    def test_end_conversation(self, chatbot):
        """Test ending a conversation"""
        chatbot.start_conversation("user_001")
        chatbot.end_conversation("user_001")
        
        # Starting a new conversation should work
        context = chatbot.get_conversation("user_001")
        assert context is not None
    
    def test_multiple_users(self, chatbot):
        """Test handling multiple concurrent users"""
        chatbot.process_message("user_001", "hello")
        chatbot.process_message("user_002", "hi there")
        
        context1 = chatbot.get_conversation("user_001")
        context2 = chatbot.get_conversation("user_002")
        
        assert context1.user_id == "user_001"
        assert context2.user_id == "user_002"
        assert context1.turn_count == 1
        assert context2.turn_count == 1
    
    def test_fallback_escalation(self, chatbot):
        """Test fallback escalation"""
        # Send multiple low-confidence messages
        for i in range(4):
            response, context = chatbot.process_message("user_001", "xyzabc123")
        
        # Should eventually escalate
        assert context.turn_count == 4


class TestIntegration:
    """Integration tests for complete workflows"""
    
    @pytest.fixture
    def chatbot(self, tmp_path):
        model_path = tmp_path / "test_model.pkl"
        chatbot = Chatbot(str(model_path))
        return chatbot
    
    def test_complete_order_workflow(self, chatbot):
        """Test complete order placement workflow"""
        conversation = [
            ("hello", "greeting"),
            ("i want to place an order", "order"),
            ("my name is Bob", None),  # Entity extraction
            ("i want a phone", None),   # Entity extraction
            ("i need 2", None),         # Entity extraction
        ]
        
        for user_msg, expected_intent in conversation:
            response, context = chatbot.process_message("user_001", user_msg)
            assert response is not None
            if expected_intent:
                assert context.current_intent == expected_intent
        
        # Check final state
        context = chatbot.get_conversation("user_001")
        assert context.turn_count == len(conversation)
    
    def test_support_workflow(self, chatbot):
        """Test customer support workflow"""
        conversation = [
            "hi",
            "my order is late",
            "it was supposed to arrive yesterday",
            "can you help",
        ]
        
        for msg in conversation:
            response, context = chatbot.process_message("user_002", msg)
            assert response is not None
        
        context = chatbot.get_conversation("user_002")
        assert context.get_context_flag("in_support_flow") is True
    
    def test_fallback_recovery_workflow(self, chatbot):
        """Test recovery from fallback"""
        # Start with unclear message
        chatbot.process_message("user_003", "blahblah")
        
        # Then provide clear message
        response, context = chatbot.process_message("user_003", "hello")
        
        assert context.current_intent == 'greeting'


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
