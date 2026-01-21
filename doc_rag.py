from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
import warnings

load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)

app = FastAPI()

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

# ---- Simple HTML page ----
HTML_FORM = """
<!DOCTYPE html>
<html>
  <head><title>Q&A</title></head>
  <body>
    <h2>Ask a question</h2>
    <form method="post">
      <input type="text" name="question" style="width:400px" required>
      <button type="submit">Ask</button>
    </form>
    {answer_block}
  </body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    history_html = "".join(
        f"<p><b>{msg['role'].capitalize()}:</b> {msg['content']}</p>"
        for msg in chat_history
    )
    return HTML_FORM.format(answer_block=history_html)


@app.post("/", response_class=HTMLResponse)
async def ask(question: str = Form(...)):
    # Store the user's message
    chat_history.append({"role": "user", "content": question})

    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(d.page_content for d in retrieved_docs)
    final_prompt = prompt.invoke({"context": context_text, "question": question})
    answer = llm.invoke(final_prompt)

    # Store the assistant's reply
    chat_history.append({"role": "assistant", "content": answer.content})

    # Build an HTML block showing the conversation
    history_html = "".join(
        f"<p><b>{msg['role'].capitalize()}:</b> {msg['content']}</p>"
        for msg in chat_history
    )
    return HTML_FORM.format(answer_block=history_html)

