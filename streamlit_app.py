from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import  HumanMessage
from langgraph.graph import add_messages, StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import ArxivLoader
import os, streamlit as st

load_dotenv()
LANGSMITH_TRACING="true"
LANGSMITH_PROJECT="RAG_app"
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING'] = "true"
os.environ['LANGSMITH_PROJECT'] = os.getenv('LANGSMITH_PROJECT')

llm = ChatGroq(model='llama-3.1-8b-instant')

hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

memory = MemorySaver()


#load paper
loader = ArxivLoader(
    query="1706.03762",     #query is paper number
    load_max_docs=1,
    # load_all_available_meta=False
)
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# indexing
vector_store = Chroma.from_documents(
    documents=all_splits, 
    embedding=hf_embeddings,
    persist_directory="./chroma_db"
)

# turning retrival step as a tool

def retrive(query: str):
    """Retrieve information related to a query."""
    retrived_docs = vector_store.similarity_search(query,k=2)
    serialized = "\n\n".join(
        (f"Souce: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrived_docs
    )
    return serialized, retrived_docs

tools = [retrive]

class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]


def generate(state: BasicChatState):
    """Generates answer"""
    llm_with_tools = llm.bind_tools(tools=tools)
    return{
        "messages": [llm_with_tools.invoke(state["messages"])]
    }

def tools_router(state: BasicChatState):
    last_message = state["messages"][-1]

    if (hasattr(last_message,"tool_calls") and len(last_message.tool_calls)>0):
        return "tool_node"
    else:
        return END
    
tool_node = ToolNode(tools=tools,messages_key="messages")

graph = StateGraph(BasicChatState)
graph.add_node("generate", generate)
graph.add_node("tool_node",tool_node)
graph.set_entry_point("generate")
graph.add_conditional_edges("generate",tools_router)
graph.add_edge("tool_node","generate")

app = graph.compile(checkpointer=memory)

# Streamlit UI
st.set_page_config(page_title="RAG Paper QA", layout="centered")
st.title("📄 Arxiv Paper Chatbot")

with st.chat_message('ai'):
    st.write("Hello 👋")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("What is up?")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # create unique thread ID in runtime
    if "thread_id" not in st.session_state:
        import uuid
        st.session_state.thread_id = str(uuid.uuid4())

    config = {"configurable": { "thread_id": st.session_state.thread_id }}


    result = app.invoke({
        "messages": [HumanMessage(content=user_input)]
    }, config=config)

    ai_response = result["messages"][-1].content
    with st.chat_message("assistant"):
        st.markdown(ai_response)
    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

