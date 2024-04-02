import streamlit as st
import os
from pathlib import Path
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃"
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


api_key = st.sidebar.text_input(
    "put your OpenAI API Key here", type="password")

memory_llm = None

if api_key:
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        api_key=api_key,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ]
    )
    memory_llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        api_key=api_key,
    )
    memory = ConversationSummaryBufferMemory(
        llm=memory_llm,
        max_token_limit=120,
        memory_key="chat_history",
        return_messages=True,
    )
else:
    st.warning("Please enter your OpenAI API Key first!!")


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    files_dir = Path("./.cache/files")
    os.makedirs(files_dir, exist_ok=True)
    file_path = files_dir / file.name
    with open(file_path, "wb") as f:
        f.write(file_content)
    embeddings_dir = Path("./.cache/embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    cache_dir = LocalFileStore(str(embeddings_dir / file.name))
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(api_key=api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def save_memory(input, output):
    st.session_state["chat_history"].append({"input": input, "output": output})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def restore_memory():
    for history in st.session_state["chat_history"]:
        memory.save_context(
            {"input": history["input"]},
            {"output": history["output"]}
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def load_memory(_):
    return memory.load_memory_variables({})["chat_history"]


def invoke_chain(message):
    result = chain.invoke(message)
    save_memory(message, result.content)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
        Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
        Context: {context}
        """
         ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ]
)

st.title("DocumentGPT")

st.markdown("""
            Welcome!
            
            Use this chatbot to ask questions to an AI about your files!

            Upload your files on the sidebar.
            """
            )

with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf or .docx file", type=[
                            "pdf", "txt", "docx"])

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    restore_memory()
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "chat_history": load_memory,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            invoke_chain(message)
else:
    st.session_state["messages"] = []
    st.session_state["chat_history"] = []
