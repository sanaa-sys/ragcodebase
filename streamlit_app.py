import streamlit as st
import os
from github import Github
from git import Repo
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

# Initialize Pinecone
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
pinecone_index = pc.Index("codebase-rag")

# Initialize OpenAI client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["GROQ_API_KEY"]
)

# Function to clone repository
def clone_repository(repo_url):
    repo_name = repo_url.split("/")[-1]
    repo_path = f"/tmp/{repo_name}"
    Repo.clone_from(repo_url, repo_path)
    return repo_path

# Function to get file content
def get_file_content(file_path, repo_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        rel_path = os.path.relpath(file_path, repo_path)
        return {"name": rel_path, "content": content}
    except Exception as e:
        st.error(f"Error processing file {file_path}: {str(e)}")
        return None

# Function to get main files content
def get_main_files_content(repo_path):
    SUPPORTED_EXTENSIONS = {'.py', '.js', '.tsx', '.jsx', '.ipynb', '.java', '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h'}
    IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.git', '__pycache__', '.next', '.vscode', 'vendor'}
    files_content = []
    for root, _, files in os.walk(repo_path):
        if any(ignored_dir in root for ignored_dir in IGNORED_DIRS):
            continue
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
                file_content = get_file_content(file_path, repo_path)
                if file_content:
                    files_content.append(file_content)
    return files_content

# Function to get Hugging Face embeddings
def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)

# Function to get code chunks
def get_code_chunks(file_content):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_text(file_content)

# Function to perform RAG
def perform_rag(query, namespace):
    raw_query_embedding = get_huggingface_embeddings(query)
    top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=5, include_metadata=True, namespace=namespace)
    contexts = [item['metadata']['text'] for item in top_matches['matches']]
    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query
    system_prompt = """You are a Senior Software Engineer working in Google for over 20 years.
    Answer any questions I have about the codebase, based on the code provided. Always consider all of the context provided when forming a response."""
    llm_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )
    return llm_response.choices[0].message.content

# Streamlit UI
st.title("Codebase RAG Chatbot")

# Initialize session state
if 'repo_processed' not in st.session_state:
    st.session_state.repo_processed = False
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'repo_url' not in st.session_state:
    st.session_state.repo_url = ""

# Repository URL input
repo_url = st.text_input("Enter GitHub Repository URL:", key="repo_url_input")

if repo_url and repo_url != st.session_state.repo_url:
    st.session_state.repo_url = repo_url
    st.session_state.repo_processed = False
    st.session_state.conversation_history = []

if st.session_state.repo_url:
    if not st.session_state.repo_processed:
        with st.spinner("Processing repository..."):
            try:
                # Clone repository
                repo_path = clone_repository(st.session_state.repo_url)
                
                # Get file contents
                file_content = get_main_files_content(repo_path)
                
                # Process and store embeddings
                documents = []
                for file in file_content:
                    code_chunks = get_code_chunks(file['content'])
                    for i, chunk in enumerate(code_chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={"source": file['name'], "chunk_id": i, "text": chunk}
                        )
                        documents.append(doc)
                
                vectorstore = PineconeVectorStore.from_documents(
                    documents=documents,
                    embedding=HuggingFaceEmbeddings(),
                    index_name="codebase-rag",
                    namespace=st.session_state.repo_url
                )
                
                st.session_state.repo_processed = True
                st.success("Repository processed and embeddings stored!")
            except Exception as e:
                st.error(f"An error occurred while processing the repository: {str(e)}")
                st.session_state.repo_processed = False

    # Chat interface
    if st.session_state.repo_processed:
        st.subheader("Chat with the Codebase RAG Bot:")
        
        # Display conversation history
        for i, message in enumerate(st.session_state.conversation_history):
            if i % 2 == 0:
                st.write("You: " + message)
            else:
                st.write("Bot: " + message)
        
        # User input
        user_question = st.text_input("Your question:", key="user_input")
        
        if user_question:
            st.session_state.conversation_history.append(user_question)
            with st.spinner("Generating response..."):
                try:
                    response = perform_rag(user_question, st.session_state.repo_url, st.session_state.conversation_history)
                    st.session_state.conversation_history.append(response)
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"An error occurred while generating the response: {str(e)}")

else:
    st.info("Please enter a GitHub repository URL to begin.")

# Add a button to clear the conversation history
if st.button("Clear Conversation"):
    st.session_state.conversation_history = []
    st.experimental_rerun()

