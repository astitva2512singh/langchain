import os
import time
import shutil
import streamlit as st
import openai
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load environment variables
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
else:
    from dotenv import load_dotenv
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

# Constants
FAISS_FOLDER = "faiss_index"
st.set_page_config(page_title="News Research Tool", layout="wide")
st.title("News Research Tool")
st.sidebar.title("Enter News Article URLs")

# URL input from sidebar
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    if url.strip():
        urls.append(url.strip())

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

# LLM instance
llm = ChatOpenAI(temperature=0.7, model="gpt-4.1-mini", max_tokens=500)

# ✅ Process URLs
if process_url_clicked:
    if not urls:
        st.warning("Please enter at least one valid URL.")
    else:
        try:
            main_placeholder.info("Fetching content from URLs...")
            docs = []

            for url in urls:
                try:
                    loader = WebBaseLoader(url)
                    loaded_docs = loader.load()
                    if not loaded_docs:
                        st.warning(f"No content loaded from: {url}")
                    else:
                        docs.extend(loaded_docs)
                        st.success(f"Loaded: {url}")
                except Exception as e:
                    st.error(f"Failed to load {url}: {e}")

            if not docs:
                st.error("No content could be loaded from any of the URLs.")
                st.stop()

            main_placeholder.info("Splitting text into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(docs)

            main_placeholder.info("Creating vector embeddings...")
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(chunks, embeddings)

            # Delete old index if exists
            shutil.rmtree(FAISS_FOLDER, ignore_errors=True)
            vectorstore.save_local(FAISS_FOLDER)

            main_placeholder.success("Documents processed and saved successfully!")

        except Exception as e:
            st.error(f"Error processing URLs: {str(e)}")
            st.stop()

# ✅ Ask questions
query = st.text_input("Ask a question about the articles:")
if query:
    if not os.path.exists(FAISS_FOLDER):
        st.error("Please process URLs first.")
    else:
        try:
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.load_local(
                folder_path=FAISS_FOLDER,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )

            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
            )

            result = chain({"query": query})

            st.subheader("Answer")
            st.write(result["result"])

            # Show sources
            sources = result.get("source_documents", [])
            # if sources:
            #     st.subheader("Sources")
            #     for doc in sources:
            #         st.markdown(f"- {doc.metadata.get('source', 'Unknown')}")

        except Exception as e:
            st.error(f"Error answering the query: {str(e)}")
