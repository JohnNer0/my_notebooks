"""
Streamlit GUI for Chatbot
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from chatbot import Chatbot
import json

# Page config
st.set_page_config(
    page_title="🤖 Chatbot Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .chat-message {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-size: 14px;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
    }
    .bot-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .slot-box {
        background-color: #fff3e0;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 4px solid #ff9800;
    }
    .intent-badge {
        display: inline-block;
        background-color: #4caf50;
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        margin-right: 8px;
    }
    .confidence-badge {
        display: inline-block;
        background-color: #2196f3;
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
    }
    .escalation-warning {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 12px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = Chatbot(model_path='models/intent_model.pkl')

if 'user_id' not in st.session_state:
    st.session_state.user_id = None

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'conversation_started' not in st.session_state:
    st.session_state.conversation_started = False

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    # User ID
    st.subheader("User ID")
    user_id = st.text_input("Enter User ID", value="user_001", key="user_id_input")
    
    if user_id != st.session_state.user_id:
        st.session_state.user_id = user_id
        st.session_state.messages = []
        st.session_state.conversation_started = False
    
    # New Conversation
    if st.button("🔄 Start New Conversation", use_container_width=True):
        st.session_state.chatbot.end_conversation(st.session_state.user_id)
        st.session_state.messages = []
        st.session_state.conversation_started = False
        st.rerun()
    
    st.divider()
    
    # Display current conversation state
    st.subheader("📊 Conversation State")
    if st.session_state.conversation_started:
        context = st.session_state.chatbot.get_conversation(st.session_state.user_id)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Turns", context.turn_count)
        with col2:
            st.metric("Intent", context.current_intent or "None")
        
        # Pending slots
        pending = st.session_state.chatbot.get_pending_slots(st.session_state.user_id)
        if pending:
            st.warning(f"⏳ **Pending Slots:** {', '.join(pending)}")
        else:
            st.success("✅ All required information collected!")
        
        # Check escalation
        if context.get_context_flag('needs_escalation'):
            st.error("🚨 **Escalation Needed:** Transfer to human agent")
        
        # Slot details
        with st.expander("📋 Slot Details", expanded=False):
            slot_info = {}
            for name, slot in context.slots.items():
                slot_info[name] = {
                    'value': slot.value,
                    'status': slot.status.value,
                    'attempts': slot.attempts
                }
            st.json(slot_info)
    
    st.divider()
    
    # Model Info
    st.subheader("🧠 Model Info")
    if st.session_state.chatbot.classifier.intents:
        st.write(f"**Intents:** {len(st.session_state.chatbot.classifier.intents)}")
        with st.expander("View Intents"):
            st.write(", ".join(st.session_state.chatbot.classifier.intents))
    
    if st.button("📈 Evaluate Model", use_container_width=True):
        with st.spinner("Evaluating model..."):
            st.session_state.chatbot.evaluate_model()
        st.success("Model evaluation complete!")


# Main content
st.markdown('<h1 class="main-header">🤖 Chatbot Assistant</h1>', unsafe_allow_html=True)

# Create columns for layout
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("💬 Chat")
    
    # Chat display area
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.messages:
            st.info("👋 Start a conversation! Type your message below.")
        else:
            for message in st.session_state.messages:
                if message['role'] == 'user':
                    st.markdown(
                        f'<div class="chat-message user-message"><b>You:</b> {message["content"]}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="chat-message bot-message"><b>Bot:</b> {message["content"]}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Show intent and confidence if available
                    if 'metadata' in message:
                        meta = message['metadata']
                        col_intent, col_conf = st.columns([1, 1])
                        with col_intent:
                            if 'intent' in meta:
                                st.markdown(
                                    f'<span class="intent-badge">Intent: {meta["intent"]}</span>',
                                    unsafe_allow_html=True
                                )
                        with col_conf:
                            if 'confidence' in meta:
                                st.markdown(
                                    f'<span class="confidence-badge">Confidence: {meta["confidence"]:.2%}</span>',
                                    unsafe_allow_html=True
                                )
    
    st.divider()
    
    # Input area
    col_input, col_send = st.columns([5, 1])
    
    with col_input:
        user_input = st.text_input(
            "Type your message here...",
            key="message_input",
            placeholder="e.g., 'hello', 'i want to order', 'help me'"
        )
    
    with col_send:
        send_button = st.button("Send", use_container_width=True, key="send_btn")
    
    # Process message
    if send_button and user_input:
        st.session_state.conversation_started = True
        
        # Get chatbot response
        with st.spinner("Processing..."):
            response, context = st.session_state.chatbot.process_message(
                st.session_state.user_id,
                user_input
            )
        
        # Add to message history
        st.session_state.messages.append({
            'role': 'user',
            'content': user_input
        })
        
        st.session_state.messages.append({
            'role': 'bot',
            'content': response,
            'metadata': {
                'intent': context.current_intent,
                'confidence': st.session_state.chatbot.classifier.predict(user_input)['confidence']
            }
        })
        
        st.rerun()

with col2:
    st.subheader("📈 Analytics")
    
    if st.session_state.conversation_started:
        context = st.session_state.chatbot.get_conversation(st.session_state.user_id)
        summary = st.session_state.chatbot.get_conversation_summary(st.session_state.user_id)
        
        # Summary card
        with st.container():
            st.metric("Total Messages", len(st.session_state.messages))
            st.metric("Turns", summary['turn_count'])
            
            if summary['unfilled_slots']:
                st.warning(f"Missing: {', '.join(summary['unfilled_slots'])}")
            else:
                st.success("All slots filled")
            
            # Context flags
            with st.expander("Context Flags", expanded=False):
                if summary['context_flags']:
                    for flag, value in summary['context_flags'].items():
                        st.write(f"{flag}: {value}")
                else:
                    st.write("No flags set")
    else:
        st.info("👈 Start a conversation to see analytics")

# Footer
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🤖 Powered by scikit-learn")
with col2:
    st.caption("Built with Streamlit")
with col3:
    st.caption("v1.0.0")
