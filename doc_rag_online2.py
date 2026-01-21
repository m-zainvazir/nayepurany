from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer


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

Store Context:
{context}

Customer Question:
{question}

""",

    input_variables=["context", "question"],
)


# ---- Simple HTML page ----
# Replace the HTML_FORM with this fixed version

HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>Thrift Store Assistant</title>
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
            justify-content: center; /* Center the chat box vertically */
        }}
        
        .chat-container {{
            flex: 1;
            display: flex;
            flex-direction: column;
            max-width: 800px;
            margin: 20px auto; /* Center horizontally and add margin */
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
            max-height: calc(100vh - 40px); /* Limit height on small screens */
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
            overflow-y: auto; /* Enables scrolling for messages */
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
                 // Ensure we only try to scroll if the element exists
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }}
        }}

        window.onload = function() {{
            let userId = getCookie('user_id');
            if (!userId) {{
                userId = 'user_' + Math.random().toString(36).substr(2, 9);
                setCookie('user_id', userId, 30);
            }}
            document.getElementById('userId').value = userId;
            // Scroll to the bottom on page load (after a message is sent)
            scrollToBottom();
        }}
    </script>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h2>🛍️ Thrift Store Assistant</h2>
            <p>Ask me about clothing, products, or shopping help!</p>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            {answer_block}
        </div>
        
        <div class="chat-input-container">
            <form method="post" class="input-group">
                <input type="hidden" id="userId" name="user_id">
                <input type="text" name="question" class="chat-input" placeholder="Type your question about thrift items..." required>
                <button type="submit" class="send-button">Send</button>
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
async def ask(question: str = Form(...), user_id: str = Form(...)):
    # Retrieve or create user history
    if user_id not in user_histories:
        user_histories[user_id] = []
    user_chat_history = user_histories[user_id]

    # Store the user's message
    user_chat_history.append({"role": "user", "content": question})

    # Retrieve relevant context and generate answer
    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(d.page_content for d in retrieved_docs)
    final_prompt = prompt.invoke({"context": context_text, "question": question})
    answer = llm.invoke(final_prompt)

    # Store assistant's reply
    user_chat_history.append({"role": "assistant", "content": answer.content})

    # Build conversation HTML for this specific user, using the new CSS classes
    # THIS BLOCK IS NOW CORRECTLY INDENTED inside the function
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

