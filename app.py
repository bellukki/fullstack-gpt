import json
import streamlit as st
import os
from pathlib import Path
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 (TEN) questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor|Painter|Actor|Model
         
    Your turn!
         
    Context: {context}
                """
        )
    ]
)

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            },
        },
        "required": ["questions"],
    },
}


st.set_page_config(
    page_title="QuizGPT",
    page_icon="⁉",
)

st.title("QuizGPT")
st.sidebar.markdown("""
[💻Github repo &rarr;](https://github.com/bellukki/fullstack-gpt)  
[📜Code of app &rarr;](https://github.com/bellukki/fullstack-gpt/blob/master/app.py)
        
        """)

api_key = st.sidebar.text_input(
    "Put your OpenAI API Key here", type="password")

if api_key:
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-3.5-turbo-0125",
        streaming=True,
        api_key=api_key,
        callbacks=[StreamingStdOutCallbackHandler()],
    ).bind(
        function_call={"name": "create_quiz"},
        functions=[function],
    )
    difficulty = st.sidebar.selectbox(
        "Choose the difficulty of Quiz.",
        (
            "Easy",
            "Hard",
        ),
    )

    difficulty_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """
        You are a learning expert who adjusts the difficulty of questions. You are given 10 questions and need to adjust them according to the following conditions 
        1. the questions must be numbered from 1 to 10 from the top.2. if the difficulty level is Easy, the questions should be easy enough for elementary school students to answer.
        3. if the difficulty level is hard, the questions should be hard enough that you need to memorize them all to solve them.
        
        difficulty: {difficulty}
        """),
    ])

    difficulty_chain = difficulty_prompt | llm
else:
    st.warning("Please enter your OpenAI API Key first!!")


@st.cache_resource(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_dir = Path("./.cache/quiz_files/{file.name}")
    os.makedirs(file_dir, exist_ok=True)
    file_path = file_dir / file.name
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, difficulty):
    chain = questions_prompt | llm
    response = chain.invoke({"context": _docs})
    response_json = json.loads(
        response.additional_kwargs["function_call"]["arguments"])
    return response_json


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


with st.sidebar:
    docs = None
    topic = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt, .pdf or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)

if not docs and api_key:
    st.markdown(
        """
    Welcome to QuizGPT.

    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

    Get started by uploading a file or searching on Wikipedia in the sidebar.
        """
    )
else:
    response = run_quiz_chain(docs, topic if topic else file.name, difficulty)
    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"]for answer in question["answers"]],
                index=None,
            )
            if ({"answer": value, "correct": True} in question["answers"]):
                st.success("Correct!")
            elif value is not None:
                st.error("Wrong")
        button = st.form_submit_button()
