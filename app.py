import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
import openai as client
from dotenv import load_dotenv
import time
import json
import datetime

load_dotenv()


def get_keyword_duck(inputs):
    ddg = DuckDuckGoSearchResults()
    keyword = inputs["keyword"]
    return ddg.run(f"research of {keyword}")


def get_keyword_wiki(inputs):
    wiki = WikipediaAPIWrapper()
    keyword = inputs["keyword"]
    return wiki.run(f"research of {keyword}")


def research_overview(inputs):
    url = inputs["url"]
    headers = {
        'user-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        tags_to_search = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']
        text_content = '\n'.join([element.get_text(strip=True)
                                 for element in soup.find_all(tags_to_search)])
        return text_content
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


def save_to_file(inputs):
    data = inputs["data"]
    with open(f"search_result.txt", "w") as f:
        f.write(data)
    return f"Information saved in result.txt file."


functions_map = {
    "get_keyword_duck": get_keyword_duck,
    "get_keyword_wiki": get_keyword_wiki,
    "research_overview": research_overview,
    "save_to_file": save_to_file,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "get_keyword_duck",
            "description": "Show research results found via entered keyword and url.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "The research of keyword",
                    }
                },
                "required": ["keyword"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_keyword_wiki",
            "description": "Show research results found via entered keyword and url.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "The research of keyword",
                    },
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research_overview",
            "description": "Output highlights from a given search result via url.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of websites for output highlights",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_to_file",
            "description": "Saves the given data to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "string",
                        "description": "The data to save",
                    }
                },
                "required": ["data"],
            },
        },
    },
]

assistant_id = st.session_state.get("assistant_id", "")
assistant_name = st.session_state.get("assistant_name", "")
thread_id = st.session_state.get("thread_id", "")
run_id = st.session_state.get("run_id", "")
current_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content
    )


def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id).data
    messages.reverse()
    for message in messages:
        with st.chat_message(message.role):
            if message.content:
                st.markdown(message.content[0].text.value)
                if hasattr(message.content[0], 'text'):
                    if hasattr(message.content[0].text, 'annotations') and message.content[0].text.annotations:
                        for annotation in message.content[0].text.annotations:
                            st.markdown(f"Source: {annotation.ed}")


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(
            f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs


def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outputs
    )


st.set_page_config(
    page_title="ResearchGPT",
    page_icon="🕵️‍♂️",
)


st.sidebar.markdown("""
[💻Github repo &rarr;](https://github.com/bellukki/fullstack-gpt)  
[📜Code of app &rarr;](https://github.com/bellukki/fullstack-gpt/blob/researchGPT/app.py)
        """)
api_key = st.sidebar.text_input(
    "Put your OpenAI API Key here", type="password")
if api_key == "":
    st.warning("Please enter your OpenAI API Key first!!")
else:
    client = client.Client(api_key=api_key)
    if assistant_id == "":
        my_assistants = client.beta.assistants.list(
            order="desc",
            limit="10",
        )
        for assistant in my_assistants.data:
            if assistant.name == "Research AI Assistant":
                st.write(f"Found assistant: {assistant.name}, {assistant.id}")
                assistant_id = assistant.id
                assistant_name = assistant.name
                st.session_state["assistant_id"] = assistant_id
                st.session_state["assistant_name"] = assistant_name
                break

        if assistant_id == "":
            assistant = client.beta.assistants.create(
                name="Research AI Assistant",
                instructions="""
                    You're an academic specializing in IT-related research materials.
                    You should be well-versed in the latest issues and trends in IT and security. 
                                                        
                    Present as much academic research as possible for the given keywords, and be sure to provide URL sources and citations. 

                    Without the "URL" it won't be found, so be sure to include it. 
                                                    
                    You must use all three tools provided (DuckDuckGo search tool, Wikipedia search tool, and ResearchOverview tool) and include both the search results and the extracted page, as well as the Wikipedia results.
                                                        
                    Be as detailed as possible in your answer, show it to us, and save it as a txt file when you're done. Be sure to provide URL sources and citations.
                    """,
                model="gpt-4-turbo-preview",
                tools=functions,
            )
            st.session_state["assistant_id"] = assistant.id
            st.session_state["assistant_name"] = assistant.name
            assistant_id = assistant.id
            assistant_name = assistant.name
            st.write(f"Created a new assistant! ID: {assistant_id}")

    st.markdown(
        """
    # ResearchGPT

    Welcome to ResearchGPT.

    Enter keywords for the research results you want to find.
    """
    )

    if run_id != "":
        run = get_run(run_id, thread_id)
        if run.status == "completed":
            st.success("The search is completed. The details are as follows")
            get_messages(thread_id)
            with open("search_result.txt", "rb") as file:
                btn = st.download_button(
                    label="Download result.txt",
                    data=file,
                    file_name=f"{current_date}_result.txt",
                    mime="text/plain",
                    on_click=lambda: setattr(
                        st.session_state, "button_clicked", True),
                )
        elif run.status == "in_progress":
            with st.status("In progress..."):
                st.write("Waiting for the AI to respond...")
                time.sleep(3)
                st.rerun()
        elif run.status == "requires_action":
            with st.status("Processing action..."):
                submit_tool_outputs(run_id, thread_id)
                time.sleep(3)
                st.rerun()

    if "clear" in st.session_state and st.session_state["clear"]:
        st.session_state["input"] = ""
        st.session_state["clear"] = False

    keyword = st.text_input("Enter a research subject.",
                            value=st.session_state.get("input", ""),
                            key="input",)

    if keyword:
        if thread_id == "":
            thread = client.beta.threads.create(
                messages=[{"role": "user", "content": keyword}]
            )
            thread_id = thread.id
            st.session_state["thread_id"] = thread_id
        else:
            send_message(thread_id, keyword)

        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )
        run_id = run.id
        st.session_state["run_id"] = run_id
        st.session_state["clear"] = True
        st.session_state["keyword"] = keyword
        st.rerun()
