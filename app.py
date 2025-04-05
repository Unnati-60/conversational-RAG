from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import add_messages, StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, List, Annotated
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import ArxivLoader
import bs4,os

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

config = {"configurable":{
    "thread_id": "abc124"
}}

app = graph.compile(checkpointer=memory)


while True:
    user_input = input("User: ")
    if(user_input.lower() in ["exit","end","bye"]):
        break
    result = app.invoke({
        "messages": [HumanMessage(content=user_input)]
    }, config=config)
    print("AI: " + result["messages"][-1].content)
