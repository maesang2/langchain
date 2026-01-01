import streamlit as st
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI

st.set_page_config(
    page_title="FullstackGPT Home"
)

st.markdown(
    """

# Hello!

"""
)

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    
    if api_key:
        st.success("API Key 입력 완료")
    else:
        st.warning("API Key를 입력하세요")


llm = ChatOpenAI(
    temperature=1,
    api_key=api_key,
    model_name="gpt-5-nano",
    streaming=True,
)

@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
        
    # 문서
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    # vector store
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()

    return retriever

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message":message, "role":role})

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages(
       [
        (
            "system",
            """
            Answer the question using ONLY the follwing context. If you don't know the answer
            just say you don't know. DOn'T make anything up.
            -------
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

if api_key:
    with st.sidebar:
        file = st.file_uploader("Upload a .txt .pdf or .docx file", type=[
            "pdf","txt","docx"
        ])
    text="파일을 업로드하고 파일 관련된 질문을 해주세요"
else:
    file=""
    text="OpenAi api key를 입력하세요"

st.markdown(text)


if file:
    retriever = embed_file(file)
    send_message("I'm ready!", role="ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file..")
    if message:
        send_message(message, "human")
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question":RunnablePassthrough()
        } | prompt | llm
        try:
            response = chain.invoke(message)
            send_message(response.content, "ai")
        except Exception as e:
            send_message("OpenAI API Key가 유효하지 않습니다. 다시 입력해주세요.", role="ai")
        

else:
    st.session_state["messages"]=[]