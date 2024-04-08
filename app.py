import asyncio
import sys
from langchain_community.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
from langchain.callbacks import StreamingStdOutCallbackHandler

if "win32" in sys.platform:
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    cmds = [["C:Windows/system32/HOSTNAME.EXE"]]
else:
    cmds = [
        ["du", "-sh", "/Users/fredrik/Desktop"],
        ["du", "-sh", "/Users/fredrik"],
        ["du", "-sh", "/Users/fredrik/Pictures"]
    ]

answers_prompt = ChatPromptTemplate.from_template("""
    Using ONLY the following context answer the user's question.
    If you can't just say you don't know, don't make anything up.
                                                                             
    Then, give a score to the answer between 0 and 5.
    The score should be high if the answer is related to the user's question, and low otherwise.
                                                  
    If there is no relevant content, the score is 0.
    Always provide scores with your answers
    Context: {context}

    Examples:

    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!
    Question: {question}
""")


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


st.title("SiteGPT")
st.sidebar.markdown("""
[💻Github repo &rarr;](https://github.com/bellukki/fullstack-gpt)  
[📜Code of app &rarr;](https://github.com/bellukki/fullstack-gpt/blob/SiteGPT/app.py)
        """)

url_cloudflare = "https://developers.cloudflare.com/sitemap.xml"
url_openAI = "https://openai.com/sitemap.xml"

api_key = st.sidebar.text_input(
    "Put your OpenAI API Key here", type="password")

if api_key:
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        api_key=api_key,
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    st.markdown(
        """
        Ask questions about the content of a website.
                
        Start by writing the URL of the website on the sidebar.

        Or select the SITE what you want to find.

        If you are forced to disconnect, please wait 3 seconds and select again.
    """
    )
else:
    st.warning("Please enter your OpenAI API Key first!!")


def get_answers(inputs):
    docs = inputs['docs']
    question = inputs['question']
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers":
        [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"]
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"for answer in answers)
    return choose_chain.invoke({
        "question": question,
        "answers": condensed
    })


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("closeSearch Submit Blog", "")
    )


@st.cache_resource(show_spinner="Loading Website...")
def load_website(url, api_key):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200
    )
    loader = SitemapLoader(
        url,
        parsing_function=parse_page
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


@st.cache_resource(show_spinner="Loading Website...")
def load_cfwebsite(url, api_key):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200
    )
    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/workers-ai\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/ai-gateway\/).*",
        ],
        parsing_function=parse_page
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


with st.sidebar:
    choice = st.selectbox(
        "Choose what you want to find documentation.",
        (
            "Search for what you want",
            "Cloudflare",
            "OpenAI",
        ),
    )
    if choice == "Cloudflare":
        url = url_cloudflare
    elif choice == "OpenAI":
        url = url_openAI
    else:
        url = st.text_input(
            "Write down a URL",
            placeholder="https://example.com/sitemap.xml",
        )
if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Plaese write down a Sitemap URL")
    elif url == url_cloudflare:
        retriever = load_cfwebsite(url, api_key)
        query = st.text_input(
            """
            Ask a question to the Cloudflare's each one of these products:
            - AI Gateway
            - Cloudflare Vectorize
            - Workers AI
            """)
        if query:
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            result = chain.invoke(query)
            st.markdown(result.content.replace("$", "\$"))
    else:
        retriever = load_website(url, api_key)
        query = st.text_input("Ask a question to the website.")
        if query:
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            result = chain.invoke(query)
            st.markdown(result.content.replace("$", "\$"))
