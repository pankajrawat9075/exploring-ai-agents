# get the keys

import os 
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = ''
os.environ["GOOGLE_API_KEY"] = ""

# import important libraries
import requests
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

# load the pdf and split it to chunks
current_directory = os.getcwd()
print(f"current directory: {current_directory}")
file_path = (
    os.path.join(current_directory, "data", "iesc111.pdf")
)

loader = PyPDFLoader(file_path)
pages = loader.load_and_split()


# build a vector database and it's retriever
vector_store = FAISS.from_documents(pages, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
retriever = vector_store.as_retriever(k = 5)

# build the prompt. It will include "context" from retriever, "chat_history" and question from user

template = """You are a helpful assistant.
    Use the retriever_tool for context if asked question on topic sound. Say don't know the answer if you can't find the answer. 
    Also you can use youtube video links finder tool. So when asked about videos or links about topic, 
    use the tool and return the links as well. You can use both the tools if required.
    
    The chat history so far is provided.
    chat history:
    {chat_history}"""

messages = [SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], input_types={}, partial_variables={}, template=template), additional_kwargs={}),
   HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], input_types={}, partial_variables={}, template='{input}'), additional_kwargs={}),
   MessagesPlaceholder(variable_name='agent_scratchpad')]

prompt = ChatPromptTemplate.from_messages(
        messages

)

# build retriever tool
retriever_tool = create_retriever_tool(
    retriever,
    "sound_search",
    "Search for information about 'sound'. For any questions about 'sound', you must use this tool!",
)


# youtube tool
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent

# Set your YouTube Data API key
YOUTUBE_API_KEY = "AIzaSyAFmHjkY8q1c-BFC8_d9645GtigoIZAJ3E"  # Ensure you set this in your environment variables

@tool
def search_youtube(query: str):
    """Search YouTube for videos related to the query asked."""
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=5&q={query}&key={YOUTUBE_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        results = response.json()
        video_links = []
        for item in results.get("items", []):
            video_id = item["id"].get("videoId")
            if video_id:
                video_links.append(f"https://www.youtube.com/watch?v={video_id}")
        return video_links
    else:
        return ["Error retrieving videos. Please try again later."]



tools = [retriever_tool, search_youtube]
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def format_history_to_str(history_list):
    history = ""
    for chat in history_list:
        history += f"User: {chat[0]}\nAI: {chat[1]}\n"

    return history


def ask_question(question, chat_history):
    
    print(chat_history)
    answer = agent_executor.invoke({"input": question, "chat_history": format_history_to_str(chat_history)})
    
    return {"answer": answer["output"], "chat_history": "nothing"}


