import os
import git
from langchain.vectorstores import FAISS
from utils import OpenAIEmbeddings
from utils import ChatOpenAI
import logging

from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
DEFAULT_CLONE_DIR = "cloned_repo"
OUTPUT_FILE = "combined_code.txt"
ALLOWED_EXTENSIONS = {".py", ".js", ".css", ".txt"}  # Adjusted based on reference notebook

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
    List all files in a directory recursively with allowed extensions.
    """
    files_list = []
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in ALLOWED_EXTENSIONS):
                    files_list.append(os.path.join(root, file))
    except Exception as e:
        logging.error(f"Error listing files: {e}")
    return files_list

def prepare_documents(input_file):
    """
    Prepare documents from a text file for RAG.
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            return [f.read()]
    except Exception as e:
        logging.error(f"Error preparing documents: {e}")
        return []

def create_embeddings(documents, api_key):
    """
    Create embeddings for documents and return a FAISS vector store.
    """
    try:
        embeddings = OpenAIEmbeddings(course_api_key=api_key)
        db = FAISS.from_texts(documents, embeddings)
        db.save_local("faiss_db")
        logging.info("FAISS vector store created and saved locally.")
        return db
    except Exception as e:
        logging.error(f"Error creating embeddings: {e}")
        return None

def create_qa_chain(retriever, llm):
    """
    Create a QA chain using LangChain components.
    """
    template = """
    Answer the question below. All questions are based on
    the following github repo {context}.
    Be as specific as possible.
    Question: {question}
    Your answers should be in the language of the question.
    """

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])


    chain = RunnableMap({
    "context": retriever | format_docs,  # Get context and format it
    "question": RunnablePassthrough()  # Pass the question unchanged
}) | prompt | llm | StrOutputParser()

    return chain

# Main Workflow

def main():
    repo_url = ''
    api_key = ''

    # Step 1: Clone repository
    clone_dir = clone_repo(repo_url)
    if not clone_dir:
        return

    # Step 2: List all files in the cloned directory
    logging.info("Listing all files in the cloned repository:")
    files = list_files(clone_dir)
    for file in files:
        logging.info(file)

    # Step 3: Combine files into a single document (simple concatenation for testing)
    combined_content = ""
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                combined_content += f"\n=== {file_path} ===\n"
                combined_content += f.read()
        except Exception as e:
            logging.error(f"Failed to read {file_path}: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
        output_file.write(combined_content)

    # Step 4: Prepare documents
    documents = prepare_documents(OUTPUT_FILE)

    # Step 5: Create FAISS embeddings
    db = create_embeddings(documents, api_key)
    if not db:
        return

    # Step 6: Set up retriever and QA chain
    retriever = db.as_retriever()
    llm = ChatOpenAI(temperature=0.0, course_api_key=api_key)
    qa_chain = create_qa_chain(retriever, llm)

    # Step 7: Ask example questions
    example_questions = [
        "сколько ручек на бекенде?",
        "На чем реализован фронтенд проекта?",
        "Есть ли авторизация в проекте?"
    ]

    for question in example_questions:
        try:
            answer = qa_chain.invoke(question)
            logging.info(f"Q: {question}\nA: {answer}")
        except Exception as e:
            logging.error(f"Error answering question '{question}': {e}")

if __name__ == "__main__":
    main()









import os
import git
import streamlit as st
from langchain.vectorstores import FAISS
from utils import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
DEFAULT_CLONE_DIR = "cloned_repo"
OUTPUT_FILE = "combined_code.txt"
ALLOWED_EXTENSIONS = {".py", ".js", ".css", ".txt"}  # Adjusted based on reference notebook

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
    List all files in a directory recursively with allowed extensions.
    """
    files_list = []
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in ALLOWED_EXTENSIONS):
                    files_list.append(os.path.join(root, file))
    except Exception as e:
        logging.error(f"Error listing files: {e}")
    return files_list

def prepare_documents(input_file):
    """
    Prepare documents from a text file for RAG.
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            return [f.read()]
    except Exception as e:
        logging.error(f"Error preparing documents: {e}")
        return []

def create_embeddings(documents, api_key):
    """
    Create embeddings for documents and return a FAISS vector store.
    """
    try:
        embeddings = OpenAIEmbeddings(course_api_key=api_key)
        db = FAISS.from_texts(documents, embeddings)
        db.save_local("faiss_db")
        logging.info("FAISS vector store created and saved locally.")
        return db
    except Exception as e:
        logging.error(f"Error creating embeddings: {e}")
        return None

def create_qa_chain(retriever, llm):
    """
    Create a QA chain using LangChain components.
    """
    template = """
    Answer the question below. All questions are based on
    the following github repo {context}.
    Be as specific as possible.
    Question: {question}
    Your answers should be in the language of the question.
    """

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    chain = RunnableMap({
        "context": retriever | format_docs,  # Get context and format it
        "question": RunnablePassthrough()  # Pass the question unchanged
    }) | prompt | llm | StrOutputParser()

    return chain

# Streamlit Application
st.title("GitHub Repository Explorer with RAG")

repo_url = st.text_input("Enter GitHub Repository URL:", placeholder="https://github.com/user/repo.git")
api_key = st.text_input("Enter OpenAI API Key:", type="password")

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
                st.write("### Select Files for RAG Database:")
                files = list_files(clone_dir)

                # Save files to session state
                if "file_status" not in st.session_state:
                    st.session_state["file_status"] = {file: True for file in files}  # Default: all selected

if "file_status" in st.session_state:
    st.write("### Select Files for RAG Database:")
    updated_file_status = {}
    for file, selected in st.session_state["file_status"].items():
        updated_file_status[file] = st.checkbox(file, value=selected, key=file)
    st.session_state["file_status"] = updated_file_status  # Update session state

if st.button("Setup RAG System"):
    if not api_key:
        st.error("Please provide an API key.")
    else:
        with st.spinner("Setting up RAG system..."):
            # Combine selected files into a single document
            selected_files = [file for file, selected in st.session_state["file_status"].items() if selected]
            combined_content = ""
            for file_path in selected_files:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        combined_content += f"\n=== {file_path} ===\n"
                        combined_content += f.read()
                except Exception as e:
                    logging.error(f"Failed to read {file_path}: {e}")

            with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
                output_file.write(combined_content)

            # Prepare documents and create embeddings
            documents = prepare_documents(OUTPUT_FILE)
            db = create_embeddings(documents, api_key)
            if not db:
                st.error("Failed to create FAISS embeddings.")
                st.stop()

            retriever = db.as_retriever()
            llm = ChatOpenAI(temperature=0.0, course_api_key=api_key)
            qa_chain = create_qa_chain(retriever, llm)
            st.session_state["qa_chain"] = qa_chain
            st.success("RAG system is ready!")

if "qa_chain" in st.session_state:
    st.header("Ask Questions")
    question = st.text_input("Enter your question about the repository:")
    if st.button("Get Answer"):
        if question:
            with st.spinner("Generating answer..."):
                try:
                    answer = st.session_state["qa_chain"].invoke(question)
                    st.success(f"Answer: {answer}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.error("Please enter a question.")
