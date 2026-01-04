from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import BaseOutputParser
import json
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```","").replace("json","")
        return json.loads(text)

output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    
    if api_key:
        st.success("API Key 입력 완료")
    else:
        st.warning("API Key를 입력하세요")


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
            }
        },
        "required": ["questions"],
    },
}

llm = ChatOpenAI(
    temperature=1,
    # api_key=api_key,
    model="gpt-5-nano",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
).bind(
    function_call={
        "name": "create_quiz",
    },
    functions=[
        function,
    ],
)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

question_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
    
    Difficulty level: {difficulty}

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
""",
            )
        ]
    )
     
questions_chain = {
        "context": format_docs
    } | question_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.
     
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
     
    Example Input:

    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
     
    Example Output:
     
    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }},
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }},
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }},
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }},
                ]
            }}
        ]
     }}
    ```
    Your turn!

    Questions: {context}

""",
        )
    ]
)

formatting_chain = formatting_prompt | llm

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
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

@st.cache_data(show_spinner="Making quiz..")
def run_quiz_chain(_docs, topic, difficulty):
        chain = {
            "context": format_docs,
            "difficulty": lambda x: difficulty
        } | question_prompt | llm | JsonOutputFunctionsParser()

        return chain.invoke(_docs)

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5, lang="ko")
    docs = retriever.invoke(term)
    return docs

with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
            st.write(docs)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    st.write("문제 난이도를 선택하세요.")
    
    if "quiz_state" not in st.session_state:
        st.session_state.quiz_state = {
            "selected_button": None,
            "selected_button_nm": None,
            "questions": None,
            "is_submitted": False,
            "is_completed": False,
            "user_answers": {}
        }

    state = st.session_state.quiz_state

    col1, col2 = st.columns(2)

    with col1:
        if st.button("어려움", type="primary", key="btn1"):
            st.session_state.quiz_state = {
                "selected_button": "btn1",
                "selected_button_nm": "어려움",
                "difficulty": "hard",
                "questions": None,
                "is_submitted": False,
                "is_completed": False,
                "user_answers": {}
            }
            st.rerun()

    with col2:
        if st.button("쉬움", type="secondary", key="btn2"):
            st.session_state.quiz_state = {
                "selected_button": "btn2",
                "selected_button_nm": "쉬움",
                "difficulty": "easy",
                "questions": None,
                "is_submitted": False,
                "is_completed": False,
                "user_answers": {}
            }
            st.rerun()


    if state["selected_button"] and state["questions"] is None:
        difficulty = "Make questions challenging and complex for advanced learners." if state.get("difficulty") == "hard" else "Make questions simple and straightforward for beginners."
        response = run_quiz_chain(docs, topic if topic else file.name, difficulty)
        state["questions"] = response["questions"]

    if state["selected_button_nm"] is not None:
        st.write(f"선택 난이도 {state['selected_button_nm']}")

    if state["questions"] and not state["is_completed"]:
        with st.form("questions_form"):
            for idx, question in enumerate(state["questions"]):
                if state["is_submitted"]:
                    user_answer = state["user_answers"].get(idx)
                    is_correct = {"answer": user_answer, "correct": True} in question["answers"] if user_answer else False
                    color = "green" if is_correct else "red"
                    st.markdown(f"### :{color}[문제 {idx + 1}]")
                else:
                    st.markdown(f"### 문제 {idx + 1}")
                
                st.write(question["question"])
                
                value = st.radio(
                    "답을 선택하세요",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                    disabled=state["is_submitted"],
                    key=f"q_{idx}"
                )
                
                if value:
                    state["user_answers"][idx] = value
                
                if state["is_submitted"]:
                    is_correct = any(a.get("answer") == value and a.get("correct") for a in question["answers"])
    
                    if is_correct:
                        st.success("✅ 정답입니다!")
                    elif value is not None:
                        correct_answer = next((a.get("answer") for a in question["answers"] if a.get("correct")), "정답 없음")
                        st.error(f"❌ 오답입니다. 정답: **{correct_answer}**")
                    else:
                        st.warning("⚠️ 답을 선택하지 않았습니다.")
                
                st.divider()

            button = st.form_submit_button("제출하기", disabled=state["is_submitted"])

        if button and not state["is_submitted"]:
            state["is_submitted"] = True
            
            correct_count = 0
            for idx, question in enumerate(state["questions"]):
                user_answer = state["user_answers"].get(idx)
                if {"answer": user_answer, "correct": True} in question["answers"]:
                    correct_count += 1
            
            st.rerun()

        if state["is_submitted"]:
            correct_count = sum(
                1 for idx, question in enumerate(state["questions"])
                if {"answer": state["user_answers"].get(idx), "correct": True} in question["answers"]
            )
            
            st.markdown("---")
            if correct_count == len(state["questions"]):
                st.balloons()
                st.success(f"🎉 완벽해요! 모든 문제를 맞추셨습니다! ({correct_count}/{len(state['questions'])})")
                state["is_completed"] = True
            else:
                st.error(f"결과: {correct_count}/{len(state['questions'])} 정답")
                
                if st.button("🔄 다시 풀기", key="retry"):
                    st.session_state.quiz_state = {
                        "selected_button": state["selected_button"],
                        "questions": None,
                        "is_submitted": False,
                        "is_completed": False,
                        "user_answers": {}
                    }
                    st.rerun()

    elif state["is_completed"]:
        st.success("✅ 이미 완료한 퀴즈입니다! 다른 난이도를 선택해주세요.")