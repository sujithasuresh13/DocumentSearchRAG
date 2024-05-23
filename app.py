import streamlit as st
import openai
import faiss
import numpy as np

# Set your OpenAI API key
api_key = st.sidebar.text_input('Enter your OpenAI API key: ', type='password')

openai.api_key = api_key

# Function to chunk text
def chunk_text(text, chunk_size=200):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Function to get embeddings
def get_embeddings(texts):
    response = openai.Embedding.create(input=texts, model="text-embedding-ada-002")
    embeddings = [e['embedding'] for e in response['data']]
    return embeddings

# Function to index chunks using FAISS
def index_chunks(chunks):
    embeddings = get_embeddings(chunks)
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings

# Function to retrieve relevant chunks
def retrieve_relevant_chunks(query, index, chunks, k=5):
    query_embedding = get_embeddings([query])[0]
    D, I = index.search(np.array([query_embedding]), k)
    return [chunks[i] for i in I[0]]

# Few-shot examples
few_shot_examples = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "Who wrote 'To Kill a Mockingbird'?"},
    {"role": "assistant", "content": "Harper Lee wrote 'To Kill a Mockingbird'."}
]

# Function to generate a response
def generate_response(query, few_shot_examples, index, chunks):
    relevant_chunks = retrieve_relevant_chunks(query, index, chunks)
    context = "\n\n".join(relevant_chunks)
    messages = few_shot_examples + [
        {"role": "system", "content": "Use the following context to answer the question: " + context},
        {"role": "user", "content": query}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )
    return response['choices'][0]['message']['content']

# Streamlit application
st.title("Retrieval-Augmented Generation (RAG) with OpenAI")
st.write("Upload a text document to build the context database.")

# Upload file
uploaded_file = st.file_uploader("Choose a file", type=["txt"])
if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    chunks = chunk_text(text)
    index, embeddings = index_chunks(chunks)
    st.success("Document indexed successfully!")

    query = st.text_input("Enter your query:")
    if query:
        response = generate_response(query, few_shot_examples, index, chunks)
        st.write("Response:")
        st.write(response)
