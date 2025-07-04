import os
import re
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings  # Corrected Import
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import InferenceClient
from typing import Any, Dict

# Disable TensorFlow oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Custom LLM Wrapper for Hugging Face InferenceClient
class HuggingFaceInferenceWrapper:
    def __init__(self, model_id: str, token: str):
        self.client = InferenceClient(model=model_id, token=token)

    def invoke(self, prompt: str, temperature=0.5, max_new_tokens=512) -> str:
        response = self.client.text_generation(
            prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        return response.strip()  # Ensure clean response

# Load environment variables
load_dotenv(find_dotenv())

# Hugging Face API Token
HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Initialize LLM with InferenceClient
llm = HuggingFaceInferenceWrapper(model_id=HUGGINGFACE_REPO_ID, token=HF_TOKEN)

# Custom Prompt Template
custom_prompt_template = PromptTemplate(
    template="""Use the provided context to answer the query. If you don't know the answer, just say you don't know. Do not make up an answer or provide anything outside the context.

Context: {context}
Query: {query}

Provide a direct answer. If the response includes a medicine name, ensure to append the warning: 'I am not a doctor. Please consult a medical professional before taking any medicine.'
""",
    input_variables=["context", "query"],
)

# Load FAISS Vectorstore
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Define a Function to Format Retrieved Documents
def format_docs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    context = "\n".join([doc.page_content for doc in inputs["context"]]) if inputs["context"] else "No relevant context found."
    return {"context": context, "query": inputs["query"]}

# Function to Append Doctor Warning for Medicines
def add_medicine_warning(response: str) -> str:
    medicine_pattern = re.compile(r"\\b(aspirin|acetaminophen|tylenol|ibuprofen|advil|paracetamol|amoxicillin|ciprofloxacin|dolo|metformin|azithromycin|cetirizine|antibiotic|painkiller|fever medicine)\\b", re.IGNORECASE)
    if medicine_pattern.search(response):
        response = re.sub(medicine_pattern, "[REDACTED]", response)
        response += "\nI am not a doctor. Please consult a medical professional before taking any medicine."
    return response

# Create Retrieval Chain
retriever = db.as_retriever(search_kwargs={"k": 3})
retrieval_chain = (
    RunnableParallel({"context": retriever, "query": RunnablePassthrough()})
    | RunnableLambda(format_docs)
    | (lambda inputs: custom_prompt_template.format(**inputs))  # Ensure formatted string output
    | (lambda prompt: llm.invoke(prompt))  # Invoke LLM
    | RunnableLambda(lambda response: add_medicine_warning(response))  # Append doctor warning if needed
)

# Get User Query
user_query = input("Write Your Query: ").strip()
if not user_query:
    raise ValueError("User query cannot be empty.")

# Invoke Retrieval Chain
try:
    response = retrieval_chain.invoke(user_query)
    print("\nRESULT:", response)
except Exception as e:
    print(f"An error occurred: {e}")