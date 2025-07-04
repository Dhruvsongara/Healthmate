import os
from flask import Flask, render_template, request, jsonify
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import InferenceClient
from typing import Any, Dict

app = Flask(__name__)

# Load environment variables
load_dotenv(find_dotenv())
HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Custom LLM Wrapper
class HuggingFaceInferenceWrapper:
    def __init__(self, model_id: str, token: str):
        self.client = InferenceClient(model=model_id, token=token)

    def invoke(self, prompt: str, temperature=0.5, max_new_tokens=512) -> str:
        response = self.client.text_generation(
            prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        return response

llm = HuggingFaceInferenceWrapper(model_id=HUGGINGFACE_REPO_ID, token=HF_TOKEN)

# Custom Prompt Template
custom_prompt_template = PromptTemplate(
    template="""Use the provided context to answer the query. If you don't know the answer, just say you don't know. Do not make up an answer.

Context: {context}
Query: {query}

Provide a direct answer.
""",
    input_variables=["context", "query"],
)

# Load FAISS Vectorstore
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

def format_docs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    context = "\n".join([doc.page_content for doc in inputs["context"]]) if inputs["context"] else "No relevant context found."
    return {"context": context, "query": inputs["query"]}

retrieval_chain = (
    RunnableParallel({"context": retriever, "query": RunnablePassthrough()})
    | RunnableLambda(format_docs)
    | (lambda inputs: custom_prompt_template.format(**inputs))
    | (lambda prompt: llm.invoke(prompt))
)

def chatbot_response(query: str):
    if not query.strip():
        return "Please enter a valid query."
    try:
        response = retrieval_chain.invoke(query)
        return response
    except Exception as e:
        return f"An error occurred: {e}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_query = request.json.get("query", "").strip()
    response = chatbot_response(user_query)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)


#http://127.0.0.1:5000/