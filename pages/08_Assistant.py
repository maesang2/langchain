import streamlit as st
from openai import OpenAI
from datetime import datetime
from bs4 import BeautifulSoup
import requests
import json
import time
from langchain_community.utilities import WikipediaAPIWrapper

st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Research Assistant")

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    
    if api_key:
        st.success("API Key 입력 완료")
    else:
        st.warning("API Key를 입력하세요")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.thread_id = None
        st.session_state.assistant_id = None
        st.rerun()

if not api_key:
    st.info("사이드바에서 OpenAI API Key를 입력해주세요.")
    st.stop()

client = OpenAI(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "assistant_id" not in st.session_state:
    st.session_state.assistant_id = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

functions = [
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Search for information on Wikipedia. Returns detailed information from Wikipedia articles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for Wikipedia"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_scraper",
            "description": "Scrape and extract text content from a website. Input should be a complete URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the website to scrape (starting with http:// or https://)"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "file_saver",
            "description": "Save research results to a .txt file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Content to save to file"
                    },
                    "filename": {
                        "type": "string",
                        "description": "Name of the file (without .txt extension)"
                    }
                },
                "required": ["content", "filename"]
            }
        }
    }
]

def wikipedia_search(query: str) -> str:
    try:
        wiki = WikipediaAPIWrapper()
        result = wiki.run(query)
        return result
    except Exception as e:
        return f"Wikipedia search error: {str(e)}"

def web_scraper(url: str) -> str:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
            tag.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        cleaned_text = '\n'.join(lines)
        
        if len(cleaned_text) > 10000:
            cleaned_text = cleaned_text[:10000] + "\n\n...(content truncated for length)"
        
        return f"Content from {url}:\n\n{cleaned_text}"
    except Exception as e:
        return f"Web scraping error for {url}: {str(e)}"

def file_saver(content: str, filename: str) -> str:
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = f"{filename}_{timestamp}.txt"
        
        with open(full_filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"✅ Successfully saved research to {full_filename}"
    except Exception as e:
        return f"File saving error: {str(e)}"

available_functions = {
    "wikipedia_search": wikipedia_search,
    "web_scraper": web_scraper,
    "file_saver": file_saver,
}

if st.session_state.assistant_id is None:
    with st.spinner("Initializing assistant..."):
        assistant = client.beta.assistants.create(
            name="Research Assistant",
            instructions="""You are a helpful research assistant that helps users gather information and save it to files.
            
            When researching:
            1. Use wikipedia_search to find information
            2. If needed, use web_scraper to get more details from specific URLs
            3. When user asks to save, use file_saver to save the research report
            
            Be concise and helpful in your responses.""",
            tools=functions,
            model="gpt-4o-mini",
        )
        st.session_state.assistant_id = assistant.id

if st.session_state.thread_id is None:
    thread = client.beta.threads.create()
    st.session_state.thread_id = thread.id

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("검색 도우미 입니다. 질문하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        client.beta.threads.messages.create(
            thread_id=st.session_state.thread_id,
            role="user",
            content=prompt
        )
        
        run = client.beta.threads.runs.create(
            thread_id=st.session_state.thread_id,
            assistant_id=st.session_state.assistant_id
        )
        
        with st.spinner("Thinking..."):
            while True:
                run = client.beta.threads.runs.retrieve(
                    thread_id=st.session_state.thread_id,
                    run_id=run.id
                )
                
                if run.status == "completed":
                    break
                elif run.status == "requires_action":
                    tool_outputs = []
                    
                    for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        if function_name in available_functions:
                            function_response = available_functions[function_name](**function_args)
                            
                            tool_outputs.append({
                                "tool_call_id": tool_call.id,
                                "output": function_response
                            })
                    
                    run = client.beta.threads.runs.submit_tool_outputs(
                        thread_id=st.session_state.thread_id,
                        run_id=run.id,
                        tool_outputs=tool_outputs
                    )
                elif run.status == "failed":
                    st.error(f"Run failed: {run.last_error}")
                    break
                
                time.sleep(1)
        
        messages = client.beta.threads.messages.list(
            thread_id=st.session_state.thread_id
        )
        
        for message in messages.data:
            if message.role == "assistant":
                full_response = message.content[0].text.value
                break
        
        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})