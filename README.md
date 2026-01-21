# NayePurany Bot - RAG-Based Chatbot with FastAPI + Shopify Integration

This repository contains a **Retrieval-Augmented Generation (RAG)** chatbot application built using **FastAPI**, **LangChain**, **FAISS**, and **Hugging Face embeddings**.  
The application provides a web-based chat interface and is designed to integrate smoothly with a Shopify website.

---

## 🚀 Features

- **RAG Implementation**
  - Uses **Hugging Face sentence-transformer embeddings** (`all-MiniLM-L6-v2`)
  - **FAISS** vector store for fast similarity search
  - Context-aware responses based on uploaded store documents

- **LLM Integration**
  - Powered by **Groq LLM (LLaMA 3.1 8B Instant)**
  - Short, concise, customer-friendly responses

- **Web Backend**
  - Built with **FastAPI**
  - Simple, responsive **HTML/CSS chat UI**
  - Cookie-based user tracking for **individual chat histories**

- **Shopify Integration Ready**
  - Can be embedded into Shopify storefronts via an iframe widget
  - Designed to run seamlessly alongside an existing e-commerce site

- **Production Deployment**
  - Hosted on a **remote Linux server**
  - Managed using **PuTTY**
  - Served using **Uvicorn**

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate    # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
Create a `.env` file and add required API keys (e.g. Groq):
```env
GROQ_API_KEY=your_api_key_here
```

---

## ▶️ Running the Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn doc_rag_final_done:app --host 0.0.0.0 --port 8000
```

Open your browser and visit:
```
http://localhost:8000
```

---

## 🧠 How It Works

1. Store documents are loaded and split into chunks
2. Chunks are embedded using **Hugging Face sentence transformers**
3. Embeddings are stored in a **FAISS vector database**
4. User queries retrieve the most relevant context
5. Context + chat history are sent to the **Groq LLM**
6. The response is displayed in the chat UI

---

## 🛍 Shopify Integration

- The chatbot can be embedded using an iframe widget endpoint
- Supports customer queries about products, thrift items, and store policies
- Cookie-based tracking ensures a smooth, personalized experience per visitor

---

## 🔐 Security Notes

- Store API keys in `.env` files (never commit them)
- Use HTTPS and a reverse proxy (e.g., Nginx) in production
- Consider persistent storage (Redis / DB) for chat history in large-scale deployments

---

## 📦 Dependencies

All dependencies are listed in `requirements.txt`, including:

- FastAPI
- Uvicorn
- LangChain
- FAISS
- Sentence Transformers
- Hugging Face Transformers
- Groq LLM SDK

---

## 📄 License

This project is intended for internal or commercial use.  
Add an open-source license (MIT / Apache 2.0) if you plan to distribute publicly.

---

## ✨ Author

Developed for a **RAG-powered Shopify chatbot** deployment using FastAPI and modern LLM tooling.
