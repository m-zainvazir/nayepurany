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

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer from the provided transcript context.
    If not provided in the context, answer from your knowledge, but mention that the answer is not in the document.
    Make your answer consize, and humanly.

    {context}
    Question: {question}
    """,
    input_variables=["context", "question"],
)

prompt = PromptTemplate(
    template="""

You are an assistant for a clothing and thrift store.
Your goal is to provide VERY SHORT, helpful responses.

STRICT RULES:
1. Answer in 1-2 short sentences MAXIMUM
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
    <title>Q&A</title>

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

    window.onload = function() {{
        let userId = getCookie('user_id');
        if (!userId) {{
        userId = 'user_' + Math.random().toString(36).substr(2, 9);
        setCookie('user_id', userId, 30);
        }}
        document.getElementById('userId').value = userId;
    }}
    </script>


  </head>
  <body>
    <h2>Ask a question</h2>
    <form method="post">
      <input type="hidden" id="userId" name="user_id">
      <input type="text" name="question" style="width:400px" required>
      <button type="submit">Ask</button>
    </form>
    {answer_block}
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

    # Build conversation HTML for this specific user
    history_html = "".join(
        f"<p><b>{msg['role'].capitalize()}:</b> {msg['content']}</p>"
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

