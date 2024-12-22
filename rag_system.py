import os
import git
from pathlib import Path
import streamlit as st
import logging
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from utils import ChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DEFAULT_CLONE_DIR = "cloned_repo"

# Helper Functions

def clone_repo(repo_url, clone_dir=DEFAULT_CLONE_DIR):
    """
    Clone a repository from a given URL to the specified directory.
    """
    try:
        if not os.path.exists(clone_dir):
            git.Repo.clone_from(repo_url, clone_dir)
            logging.info(f"Repository cloned to {clone_dir}")
        else:
            logging.info(f"Repository already exists at {clone_dir}")
        return clone_dir
    except Exception as e:
        logging.error(f"Error cloning repository: {e}")
        return None

def list_files(directory):
    """
    List all files in a directory recursively.
    """
    files_list = []
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                files_list.append(os.path.join(root, file))
    except Exception as e:
        logging.error(f"Error listing files: {e}")
    return files_list

def create_qa_chain(retriever, llm):
    """
    Create a QA chain using LangChain components.
    """
    template = """
    Use the following context to answer the question.
    Context: {context}
    Question: {question}
    Be specific and concise.
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(
        retriever=retriever,
        chain_type="stuff",
        llm=llm,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return chain

# Main Workflow

def main(repo_url, api_key, llm):
    # Clone repository
    clone_dir = clone_repo(repo_url)
    if not clone_dir:
        return

    # List all files in the cloned directory
    print("\nListing all files in the cloned repository:")
    files = list_files(clone_dir)
    for file in files:
        print(file)

    # Placeholder for QA system setup (omitted for brevity)
    documents = []  # Replace with actual document preparation logic
    retriever = None  # Replace with retriever setup logic
    qa_chain = create_qa_chain(retriever, llm)  # Replace with valid retriever

    # Example questions
    example_questions = [
        "How many endpoints are in the backend?",
        "What framework is used in the frontend?",
        "Does the project have authentication?"
    ]

    for question in example_questions:
        try:
            answer = qa_chain.run(question)
            logging.info(f"Q: {question}\nA: {answer}")
        except Exception as e:
            logging.error(f"Error answering question '{question}': {e}")

# Streamlit Application
st.title("GitHub Repository File Explorer with RAG")

# Input for repository URL
repo_url = st.text_input("Enter GitHub Repository URL:", placeholder="https://github.com/user/repo.git")
api_key = st.text_input("Enter API Key:", type="password")

if st.button("Clone and List Files"):
    if not repo_url:
        st.error("Please provide a valid GitHub repository URL.")
    else:
        with st.spinner("Cloning repository..."):
            clone_dir = clone_repo(repo_url)
            if not clone_dir:
                st.error("Failed to clone the repository. Check the URL and try again.")
            else:
                st.success("Repository cloned successfully!")

                # List files in the cloned repository
                st.write("### Files in the Repository:")
                files = list_files(clone_dir)
                if files:
                    for file in files:
                        st.write(file)
                else:
                    st.warning("No files found in the repository.")

if st.button("Setup RAG System"):
    if not api_key:
        st.error("Please provide an API key.")
    else:
        with st.spinner("Setting up RAG system..."):
            llm = ChatOpenAI(temperature=0.0, course_api_key=api_key)
            documents = []  # Replace with actual document preparation logic
            retriever = None  # Replace with retriever setup logic
            qa_chain = create_qa_chain(retriever, llm)  # Replace with valid retriever
            st.session_state["qa_chain"] = qa_chain
            st.success("RAG system is ready!")

if "qa_chain" in st.session_state:
    st.header("Ask Questions")
    question = st.text_input("Enter your question about the repository:")
    if st.button("Get Answer"):
        if question:
            with st.spinner("Generating answer..."):
                try:
                    answer = st.session_state["qa_chain"].run(question)
                    st.success(f"Answer: {answer}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.error("Please enter a question.")
