# MongoDB Chatbot API

A FastAPI-based REST API that wraps the MongoDB-Connected Chatbot functionality, enabling multiple users to interact with the chatbot simultaneously through a **single unified chat endpoint**.

## 🚀 Features

- **Unified Interface**: Everything goes through the `/chat` endpoint - just like the CLI version!
- **Multi-user Support**: Each user gets their own chatbot instance
- **Chat History**: Persistent chat history per user
- **Search Functionality**: Vector and filter-based search for music
- **Context Management**: Add search results to context for analysis
- **Audio Processing**: Load and process audio files
- **Text Processing**: Process lyrics and text content
- **Analysis Queries**: Deep reasoning with o3-mini model
- **RESTful API**: Clean, documented endpoints

## 📋 Prerequisites

- Python 3.8+
- MongoDB instance running
- OpenAI API key
- All dependencies from `requirements.txt`
- Optional but recommended: run the embeddings service in `embeddings_api.py` (used by `/load` and `/loadtext`)

## 🛠️ Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Create a `.env` file with:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   MONGO_URI=your_mongodb_connection_string
    ELEVENLABS_API_KEY=your_elevenlabs_api_key_here  # Optional
    EMBEDDINGS_API_URL=http://localhost:8001          # Optional; default shown
   ```

## 🚀 Running the API

### Start the Embeddings Service (recommended)
Start the embeddings API in a separate terminal (audio/text embeddings are delegated here):
```bash
python embeddings_api.py
# or
uvicorn embeddings_api:app --host 0.0.0.0 --port 8001 --reload
```

The embeddings API will be available at `http://localhost:8001`.

### Start the Server
```bash
python app.py
```

The API will be available at `http://localhost:8000`

### Alternative: Using Uvicorn Directly
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## 📚 API Endpoints

### 🔍 Core Endpoints

#### `GET /`
- **Description**: API information and available endpoints
- **Response**: List of all endpoints with descriptions and usage examples

#### `GET /health`
- **Description**: Health check endpoint
- **Response**: API status, timestamp, and active user count

### 💬 **Main Chat Endpoint** - `/chat`

**This is the ONLY endpoint you need for all chatbot functionality!**

#### `POST /chat`
- **Description**: **Unified chat endpoint that handles everything** - just like the CLI version!
- **Request Body**:
  ```json
  {
    "message": "Your message or command here",
    "user_id": "unique_user_identifier",
    "session_id": "optional_session_id"
  }
  ```
- **Response**: Chatbot response with category classification and additional data

## 🔧 **How It Works**

The `/chat` endpoint automatically:

1. **Classifies** your input (search, analysis, chat, help, etc.)
2. **Routes** to the appropriate handler
3. **Executes** the requested functionality
4. **Returns** a unified response

### **Natural Language Examples**

```json
// Search for songs
{"message": "find me popular songs from Sweden", "user_id": "user123"}

// Ask for analysis
{"message": "explain why this trend became popular", "user_id": "user123"}

// General chat
{"message": "Hello! How are you?", "user_id": "user123"}
```

### **Command Examples**

```json
// Add search result to context
{"message": "/add 0", "user_id": "user123"}

// Load audio file
{"message": "/load /path/to/song.mp3", "user_id": "user123"}

// Process text (either inline or with --text)
{"message": "/loadtext This is song lyrics content", "user_id": "user123"}
{"message": "/loadtext --text 'This is song lyrics content'", "user_id": "user123"}

// Clear context
{"message": "/clear", "user_id": "user123"}

// Show chat history
{"message": "/history", "user_id": "user123"}

// Get help
{"message": "/help", "user_id": "user123"}
```

## 📋 Context Management

#### `GET /context/{user_id}`
- **Description**: Get the user's current context
- **Response**: List of songs in context with trend information

#### `DELETE /user/{user_id}`
- **Description**: Remove user and clean up resources
- **Response**: Confirmation message

## 🔧 Usage Examples

### CLI Client

An interactive client mirroring the original CLI is provided in `client_cli.py`.

```
export CHAT_API_URL=http://localhost:8000
export USER_ID=cli_user
python client_cli.py
```

Use the same commands as in the original CLI (e.g., `/load`, `/loadtext`, `/search`).

### Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"
USER_ID = "my_user_123"

# Send any message - it automatically handles everything!
def chat_with_bot(message):
    response = requests.post(f"{BASE_URL}/chat", json={
        "message": message,
        "user_id": USER_ID
    })
    return response.json()

# Search for songs
result = chat_with_bot("find me some popular songs")
print(f"Response: {result['response']}")
print(f"Category: {result['category']}")

# Add to context
result = chat_with_bot("/add 0")
print(f"Context Count: {result['context_count']}")

# Ask for analysis
result = chat_with_bot("explain why this trend became popular")
print(f"Analysis: {result['response']}")

# Load audio
result = chat_with_bot("/load /path/to/song.mp3")
print(f"Load Result: {result['response']}")
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Send any message - it handles everything automatically!
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "find me popular songs", "user_id": "user123"}'

# Add to context
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "/add 0", "user_id": "user123"}'

# Ask for analysis
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "explain why this trend became popular", "user_id": "user123"}'

# Load audio file
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "/load /path/to/song.mp3", "user_id": "user123"}'

### Embeddings API (optional direct calls)

```
# Upload audio and embed
curl -X POST -F "file=@/path/to/song.mp3" -F "user_id=user123" http://localhost:8001/upload-audio

# Text embedding
curl -X POST http://localhost:8001/generate-embeddings \
  -H "Content-Type: application/json" \
  -d '{"text": "Some lyrics", "user_id": "user123"}'
```
```

## 🧪 Testing

Run the test script to verify the unified chat endpoint:

```bash
python test_api.py
```

Make sure the API server is running before executing tests.

## 🔒 Security Considerations

- **User ID Validation**: Implement proper user authentication in production
- **CORS Configuration**: Adjust CORS settings for production use
- **Rate Limiting**: Consider adding rate limiting for production
- **Input Validation**: All inputs are validated using Pydantic models

## 🚀 Production Deployment

### Using Gunicorn with Uvicorn Workers
```bash
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Environment Variables for Production
```env
OPENAI_API_KEY=your_production_key
MONGO_URI=your_production_mongodb_uri
ELEVENLABS_API_KEY=your_production_key
LOG_LEVEL=info
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📊 Monitoring

- **Health Check**: Use `/health` endpoint for monitoring
- **Active Users**: Track number of active chatbot instances
- **Error Logging**: All errors are logged with user context

## 🔄 API Versioning

Current version: `1.0.0`

Future versions will maintain backward compatibility where possible.

## 📝 Error Handling

All endpoints return appropriate HTTP status codes:
- `200`: Success
- `400`: Bad Request
- `500`: Internal Server Error

Error responses include detailed error messages and user context.

## 🎯 **Key Benefits of Unified Design**

1. **Simpler Integration**: Only one endpoint to remember
2. **Natural Flow**: Just chat naturally like with the CLI
3. **Automatic Routing**: No need to figure out which endpoint to use
4. **Consistent Interface**: Same experience across CLI and API
5. **Easier Testing**: Test all functionality through one endpoint
6. **Better UX**: Users don't need to learn multiple endpoints

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.
