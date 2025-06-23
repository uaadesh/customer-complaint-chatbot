```markdown
# 🧠 RAG-Based Complaint Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that files and retrieves customer complaints using a REST API and LLM-based reasoning with memory and knowledge integration.

---

## 🚀 Features

- 💬 Chatbot interface to file and retrieve complaints
- 🧠 Uses RAG to answer general FAQs from a knowledge base
- 🔧 Complaint creation and retrieval via API (`FastAPI`)
- 🗃️ Complaints stored in SQLite
- 🧠 Local LLM support via [Ollama](https://ollama.com/)
- 📄 Knowledge base integration via FAISS
- 🎛️ Clean and interactive UI with Streamlit

---

---

## 📦 Setup Instructions

### 1. Clone & Install

```bash
git clone https://github.com/your-repo/rag-chatbot.git
cd rag-chatbot
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
````

---

### 2. Start Backend API

```bash
uvicorn backend.main:app --reload --port 8000
```

Runs the complaint registration & retrieval API on `http://localhost:8000`.

---

### 3. Start the UI (Chatbot Interface)

```bash
streamlit run ui/app.py
```

This launches the chatbot frontend.

---

## 🤖 LLM Setup with Ollama

Install [Ollama](https://ollama.com/) if you haven’t:

```bash
# On Mac/Linux/Windows (via terminal or powershell)
curl -fsSL https://ollama.com/install.sh | sh
```

Pull the LLM model:

```bash
ollama pull llama3:8b
```

Or, for your current model:

```bash
ollama pull llama3.2:3b
```

Then start Ollama server (usually runs automatically in background):

```bash
ollama run llama3.2:3b
```

Make sure `config.py` uses:

```python
use_openai = False
HF_MODEL_NAME = "Qwen/Qwen3-8B"  # Only used if you're loading HF models manually
```

And your local LLM base URL is set in `agent.py` (default is `http://localhost:11434`).

---

## 💡 Sample Interactions

```
User: I want to register a complaint.
Bot: I'd be happy to help. May I know your name?
...
Bot: ✅ Your complaint has been registered. ID: ABC123
```

```
User: Show my complaint details for ID ABC123.
Bot: Here are the details of your complaint...
```

---

## 📚 Knowledge Base

Edit `knowledge-base/faq.txt` to change answers for general questions. The bot will use semantic search + RAG to answer based on this file.

---

## ✅ APIs Available

| Endpoint           | Method | Description              |
| ------------------ | ------ | ------------------------ |
| `/complaints`      | POST   | Register a new complaint |
| `/complaints/{id}` | GET    | Retrieve complaint by ID |

---