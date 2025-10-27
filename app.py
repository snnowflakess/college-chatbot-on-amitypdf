import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
import streamlit as st
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -------- STEP 1: Load and preprocess the document --------
def load_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

@st.cache_resource
def setup_knowledge_base(pdf_path):
    text = load_pdf_text(pdf_path)
    sentences = [s for s in text.split(".") if len(s) > 20]
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    
    return sentences, index, model

sentences, index, model = setup_knowledge_base("college_info.pdf")

## -------- STEP 2: Load LLM --------
import torch
torch.device("cpu")  # force CPU mode to avoid MPS issue

qa_model = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=-1
)

# -------- STEP 3: Streamlit UI --------
st.set_page_config(page_title="College Enquiry Chatbot (RAG)", page_icon="🎓")
st.title("🎓 College Enquiry Chatbot (LLM + RAG)")
st.write("Ask me anything about your college, courses, or facilities!")

query = st.text_input("💬 Ask a question:")

if query:
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=3)
    
    retrieved_text = " ".join([sentences[i] for i in I[0]])
    prompt = f"Answer this based on the college info: {retrieved_text}\n\nQuestion: {query}"
    
    response = qa_model(prompt, max_length=100, num_return_sequences=1)[0]["generated_text"]
    
    st.success(f"**Answer:** {response}")
