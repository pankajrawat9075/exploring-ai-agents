# get the keys

import os 
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = ''
os.environ["GOOGLE_API_KEY"] = ""

# import important libraries
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.memory import ChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.vectorstores import FAISS


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

template = """Answer the question based only on the following context. If you don't know the answer, just say that you don't know.
If required, answer the question in depth but be concise. You are helping the students with their studies. Also you are given 
previous chat history.:

Context: 
{context}

Chat History:
{chat_history}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


# build the rag chain with llm from google
from operator import itemgetter
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# chat_history = ""
def format_history_to_str(history_list):
    history = ""
    for chat in history_list:
        history += f"User: {chat[0]}\nAI: {chat[1]}\n"

    return history

def ask_question(question, chat_history):
    # question = input("ask a question on topic 'sound'. Input 'stop' to stop the chat")

    # if question == 'stop':
    #     break
    rag_chain = (
        {"context": itemgetter("question") | retriever | format_docs, "question": itemgetter("question"), "chat_history": itemgetter("chat_history")}
        | prompt
        | llm
        | StrOutputParser()
    )
    # print(chat_history)
    answer = rag_chain.invoke({"question": question, "chat_history": format_history_to_str(chat_history)})

    # chat_history.append(fQuestion: {question}\nAnswer: {answer}\n"
    # print(answer, chat_history)
    return {"answer": answer, "chat_history": "nothing"}


