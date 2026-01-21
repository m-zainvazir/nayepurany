from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
#from langchain_core.pydantic_v1 import BaseModel, Field
#from langchain_core.prompts import ChatPromptTemplate


from dotenv import load_dotenv
import warnings

load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)

app = FastAPI()

user_histories = {}
chat_history = []   # each item will be {"role": "user"/"assistant", "content": "..."}


# ---- Prepare Vector Store & LLM once at startup ----
doc = "Used_Clothing_Thrift_Stores.txt"
with open(doc, "r", encoding="utf-8") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([text])

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.save("./all-MiniLM-L6-v2")

embeddings = HuggingFaceEmbeddings(model_name="./all-MiniLM-L6-v2")

#embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)

prompt = PromptTemplate(
    template="""

You are an assistant for a clothing and thrift store.
Your goal is to provide VERY SHORT, helpful responses.

STRICT RULES:
1. Answer in 1-2 shortsentences MAXIMUM
2. Use context when relevant, otherwise use your knowledge
3. Be direct and concise, no explanations
4. If suggesting follow-ups, add ONLY ONE brief suggestion at the end

Previous Conversation:
{history}

Store Context:
{context}

Customer Question:
{question}

""",

    input_variables=["history", "context", "question"],
)


# ---- Simple HTML page ----
# Replace the HTML_FORM with this fixed version

HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>NayePurany Assistant</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        
        .chat-container {{
            flex: 1;
            display: flex;
            flex-direction: column;
            max-width: 800px;
            margin: 20px auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
            max-height: calc(100vh - 40px);
        }}

        @media (max-width: 800px) {{
             .chat-container {{
                margin: 0;
                border-radius: 0;
                max-height: 100vh;
             }}
        }}
        
        .chat-header {{
            background: #4a5568;
            color: white;
            padding: 20px;
            text-align: center;
        }}
        
        .chat-messages {{
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f7fafc;
        }}
        
        .message {{
            margin-bottom: 15px;
            padding: 12px 16px;
            border-radius: 18px;
            max-width: 80%;
            word-wrap: break-word;
            line-height: 1.4;
        }}
        
        .user-message {{
            background: #4299e1;
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 5px;
        }}
        
        .assistant-message {{
            background: white;
            color: #2d3748;
            border: 1px solid #e2e8f0;
            margin-right: auto;
            border-bottom-left-radius: 5px;
        }}
        
        .chat-input-container {{
            padding: 20px;
            background: white;
            border-top: 1px solid #e2e8f0;
        }}
        
        .input-group {{
            display: flex;
            gap: 10px;
        }}
        
        .chat-input {{
            flex: 1;
            padding: 12px 16px;
            border: 1px solid #cbd5e0;
            border-radius: 25px;
            outline: none;
            font-size: 14px;
        }}
        
        .chat-input:focus {{
            border-color: #4299e1;
        }}
        
        .send-button {{
            padding: 12px 24px;
            background: #4299e1;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }}
        
        .send-button:hover {{
            background: #3182ce;
        }}

        /* --- SIMPLE Thinking Animation --- */
        .typing-indicator {{
            background: white;
            color: #2d3748;
            border: 1px solid #e2e8f0;
            margin-right: auto;
            border-bottom-left-radius: 5px;
            padding: 12px 16px;
            border-radius: 18px;
            max-width: 80%;
            margin-bottom: 15px;
            font-style: italic;
            display: none;
        }}

        .typing-indicator span {{
            height: 8px;
            width: 8px;
            background-color: #a0aec0;
            border-radius: 50%;
            display: inline-block;
            margin: 0 2px;
            animation: bounce 1.4s infinite ease-in-out both;
        }}

        .typing-indicator span:nth-child(1) {{
            animation-delay: -0.32s;
        }}
        .typing-indicator span:nth-child(2) {{
            animation-delay: -0.16s;
        }}

        @keyframes bounce {{
            0%, 80%, 100% {{ 
                transform: scale(0);
                opacity: 0.5;
            }}
            40% {{ 
                transform: scale(1);
                opacity: 1;
            }}
        }}
        /* --- END Thinking Animation --- */
    </style>

    <script>
        function getCookie(name) {{
            let match = document.cookie.match(new RegExp('(^| )' + name + '=([^;]+)'));
            return match ? match[2] : null;
        }}

        function setCookie(name, value, days) {{
            let expires = "";
            if (days) {{
                let date = new Date();
                date.setTime(date.getTime() + (days*24*60*60*1000));
                expires = "; expires=" + date.toUTCString();
            }}
            document.cookie = name + "=" + value + expires + "; path=/";
        }}

        function scrollToBottom() {{
            const messagesContainer = document.getElementById('chatMessages');
            if (messagesContainer) {{
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }}
        }}

        /* --- SIMPLE Thinking Indicator --- */
        function showThinkingIndicator() {{
            // Just show the static typing indicator and disable button
            document.getElementById('typingIndicator').style.display = 'block';
            document.getElementById('sendButton').disabled = true;
            document.getElementById('sendButton').textContent = 'Sending...';
            scrollToBottom();
            return true; // Allow form to submit normally
        }}

        window.onload = function() {{
            let userId = getCookie('user_id');
            if (!userId) {{
                userId = 'user_' + Math.random().toString(36).substr(2, 9);
                setCookie('user_id', userId, 30);
            }}
            document.getElementById('userId').value = userId;
            scrollToBottom();
            
            // Hide typing indicator on page load (in case of page refresh)
            document.getElementById('typingIndicator').style.display = 'none';
        }}
    </script>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h2>NayePurany Assistant</h2>
            <p>Ask me about clothing, products, or shopping help!</p>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            {answer_block}
            <!-- Static typing indicator that shows during form submission -->
            <div class="typing-indicator assistant-message" id="typingIndicator">
                Thinking<span></span><span></span><span></span>
            </div>
        </div>
        
        <div class="chat-input-container">
            <form method="post" class="input-group" onsubmit="return showThinkingIndicator()">
                <input type="hidden" id="userId" name="user_id">
                <input type="text" name="question" class="chat-input" id="chatInput" placeholder="Type your question about thrift items..." required>
                <button type="submit" class="send-button" id="sendButton">Send</button>
            </form>
        </div>
    </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    # Default empty history - actual history will be loaded based on user_id from POST
    return HTML_FORM.format(answer_block="")

@app.post("/", response_class=HTMLResponse)
async def ask(question: str = Form(None), user_id: str = Form(None)):
    # If no question provided, just show the current chat history
    if not question:
        user_chat_history = user_histories.get(user_id, [])
        history_html = "".join(
            f"<div class='message {msg['role']}-message'>{msg['content']}</div>"
            for msg in user_chat_history
        )
        return HTML_FORM.format(answer_block=history_html)
    
    # Rest of your existing code for handling questions...
    if user_id not in user_histories:
        user_histories[user_id] = []
    user_chat_history = user_histories[user_id]

    user_chat_history.append({"role": "user", "content": question})

    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(d.page_content for d in retrieved_docs)
    conversation_history = "\n".join(
    f"{msg['role'].capitalize()}: {msg['content']}" 
    for msg in user_chat_history[:-1]  # Exclude current question
    )

    final_prompt = prompt.invoke({
        "history": conversation_history,
        "context": context_text, 
        "question": question
    })
    answer = llm.invoke(final_prompt)

    user_chat_history.append({"role": "assistant", "content": answer.content})

    history_html = "".join(
        f"<div class='message {msg['role']}-message'>{msg['content']}</div>"
        for msg in user_chat_history
    )
    
    return HTML_FORM.format(answer_block=history_html)

# ---- Chatbot Widget Route ----
from fastapi.responses import HTMLResponse

@app.get("/widget", response_class=HTMLResponse)
async def chatbot_widget():
    return """
    <html>
    <head>
        <style>
        #chatbot {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 350px;
            height: 450px;
            border: none;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.2);
            z-index: 9999;
        }
        </style>
    </head>
    <body>
        <iframe id="chatbot" src="/" allow="clipboard-write"></iframe>
    </body>
    </html>
    """

