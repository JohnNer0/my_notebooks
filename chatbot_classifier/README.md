# Chatbot with Sentence Classifier Architecture

A comprehensive chatbot implementation featuring intent classification, conversation context management, slot filling, and intelligent fallback mechanisms.

## Directory Structure

```
chatbot_classifier/
├── intent_classifier.py        # Intent classification model
├── conversation_context.py     # Conversation state management
├── response_generator.py       # Response generation logic
├── chatbot.py                  # Main orchestrator
├── models/                     # Trained model storage
│   └── intent_model.pkl       # Serialized classifier
└── Chatbot_Demo.ipynb         # Interactive demonstration
```

## Components

### 1. Intent Classifier (`intent_classifier.py`)
- Classifies user inputs into predefined intents
- Uses TF-IDF vectorization + Random Forest classifier
- Provides confidence scores for predictions
- Supports model serialization (save/load)

**Intents:**
- `greeting`: User greets the chatbot
- `goodbye`: User says goodbye
- `help`: User asks for help
- `order`: User wants to place an order
- `support`: User reports an issue
- `information`: User asks for information

### 2. Conversation Context (`conversation_context.py`)
- Maintains conversation state per user
- Manages slots (pending, filled, confirmed)
- Tracks conversation history
- Maintains context flags (e.g., needs_escalation)

**Slot Statuses:**
- `EMPTY`: Slot not yet requested
- `PENDING`: Slot requested but not yet filled
- `FILLED`: Slot has been filled with a value
- `CONFIRMED`: Slot value has been confirmed

**Key Methods:**
- `add_slot()`: Add a required/optional slot
- `fill_slot()`: Fill a slot with a value
- `get_unfilled_required_slots()`: Get slots needing attention
- `all_slots_filled()`: Check if all required slots are filled

### 3. Response Generator (`response_generator.py`)
- Generates contextual responses based on intent
- Handles slot requests with natural language
- Implements fallback responses for low confidence
- Escalates to human agent after max attempts
- Generates confirmations and follow-ups

**Features:**
- Template-based responses with randomization
- Intelligent fallback after 3 low-confidence messages
- Slot request with attempt tracking
- Dynamic confirmation messages

### 4. Main Chatbot (`chatbot.py`)
- Orchestrates all components
- Manages multiple conversations per user
- Implements conversation logic
- Extracts entities from user messages
- Handles intent-based flow routing

**Key Methods:**
- `process_message()`: Main entry point for user input
- `start_conversation()`: Initialize conversation with user
- `get_conversation_summary()`: Get conversation state
- `get_pending_slots()`: Get unfilled required slots

## Conversation Flow

### Order Example
```
User: Hello
Bot: Hello! How can I help you today?

User: I want to place an order
Bot: Great! I'd like to help you place an order. What product are you interested in?

User: I want a laptop
Bot: What's your name?

User: My name is John
Bot: How many would you like?

User: I need 2 please
Bot: Just to confirm, name: John, product: laptop, quantity: 2. Is that correct?
```

### Fallback Example
```
User: xyz123blah
Bot: I'm not quite sure I understand. Could you rephrase that?

User: qwerty
Bot: Sorry, I didn't catch that. Can you say it differently?

User: asdfgh
Bot: I'm having trouble understanding your request. Let me connect you with a human agent who can better assist you.
```

## Slot Management

Default slots for order flow:
- `name` (required): Customer name
- `product` (required): Product to order
- `quantity` (required): Quantity desired
- `email` (optional): Customer email

Each slot tracks:
- Current value
- Status (empty/pending/filled/confirmed)
- Number of request attempts
- Maximum allowed attempts (3)

## Fallback Strategy

1. **First Low-Confidence Message**: Generic fallback asking for clarification
2. **Second Attempt**: Another clarification request
3. **Third Attempt**: More specific help message
4. **Escalation**: Transfer to human agent

Fallback is triggered when:
- Intent confidence < 0.3
- User provides completely nonsensical input
- Same slot requested >3 times

## Usage

### Basic Usage
```python
from chatbot import Chatbot

# Initialize chatbot
chatbot = Chatbot(model_path='models/intent_model.pkl')

# Process user message
response, context = chatbot.process_message(
    user_id='user_001',
    user_message='I want to place an order'
)

print(response)
# Output: Great! I'd like to help you place an order. What product are you interested in?

# Get pending slots
print(chatbot.get_pending_slots('user_001'))
# Output: ['name', 'product', 'quantity']

# Get conversation summary
print(chatbot.get_conversation_summary('user_001'))
```

### Multi-turn Conversation
```python
messages = [
    'Hello',
    'I want to buy a laptop',
    'My name is Alice',
    'I need 2',
    'alice@example.com'
]

for msg in messages:
    response, context = chatbot.process_message('user_001', msg)
    print(f"User: {msg}")
    print(f"Bot: {response}")
    print(f"Pending: {chatbot.get_pending_slots('user_001')}")
    print()
```

## Training the Model

The model is automatically trained on first use if no saved model exists:

```python
# Retrain model with new data
chatbot.train_model()

# Evaluate model performance
chatbot.evaluate_model()
```

## Extensibility

### Add New Intents
1. Add training data to `IntentClassifier.create_training_data()`
2. Add response templates to `ResponseGenerator._load_response_templates()`
3. Add logic to `Chatbot._handle_intent_logic()`

### Add New Slots
```python
context.add_slot(
    name='phone',
    required=True,
    slot_type='phone'
)
```

### Improve Entity Extraction
Replace simple keyword matching in `Chatbot._extract_and_fill_slots()` with:
- Named Entity Recognition (NER) models
- Regular expressions
- Domain-specific extractors

## Dependencies

```
scikit-learn>=0.24.0
pandas>=1.2.0
numpy>=1.20.0
```

## Files Generated

- `models/intent_model.pkl`: Serialized classifier (created on first run)

## Future Enhancements

1. **Context Window**: Remember information from previous conversations
2. **Intent Confidence Thresholds**: Configurable per intent
3. **Custom Entity Extraction**: Support for domain-specific entities
4. **Dialogue Acts**: Add more nuanced response types
5. **Multi-language Support**: Extend to other languages
6. **Analytics**: Track conversation metrics and failure modes
7. **A/B Testing**: Test different response templates
