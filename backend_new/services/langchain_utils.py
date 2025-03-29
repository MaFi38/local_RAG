from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import os
from services.vector_store_db import vectorstore
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print("OPENAI_API_KEY: inside the langchain",OPENAI_API_KEY)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

output_parser = StrOutputParser()


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
    
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

current_dir = os.path.dirname(os.path.abspath(__file__))
prompt_file_path = os.path.join(current_dir, '..', 'prompt.txt')
try:
    with open(prompt_file_path, "r") as file:
        dynamic_prompt_text = file.read().strip()
except FileNotFoundError:
    raise Exception(f"Prompt file not found at: {prompt_file_path}")


print("Dynamic Prompt Text: ",dynamic_prompt_text)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",dynamic_prompt_text),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


def get_streaming_rag_chain(model_name="gpt-4o-mini"):
    callbacks = [StreamingStdOutCallbackHandler()]
    
    llm = ChatOpenAI(
        model=model_name,
        streaming=True,
        callbacks=callbacks
    )
    
    if model_name == "llama3.1":
        llm = ChatOllama(
            model="llama3.1:latest",
            streaming=True,
            callbacks=callbacks
        )
    elif model_name == "llama3.2":
        llm = ChatOllama(
            model="llama3.2:3b",
            streaming=True,
            callbacks=callbacks
        )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain
