import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMChain, ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import ChatMessageHistory, ConversationBufferWindowMemory
import streamlit as st
import openai

# from config_openai import *
from openai import AzureOpenAI

from azure_ai_search import get_doc_azure_ai

load_dotenv()

# importing Azure OpenAI creds
api_key = os.getenv("AZURE_OPENAI_KEY")
os.environ["OPENAI_API_KEY"] = api_key
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = "gpt"

# Load environment variables
# AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# Initialize the LLM (Language Model) with Azure OpenAI credentials
llm = AzureChatOpenAI(
    api_version="2023-12-01-preview",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    deployment_name="gpt",
    model_name="gpt-35-turbo",
    temperature=0.9,
)

client = AzureOpenAI(
    azure_endpoint=azure_endpoint, api_key=api_key, api_version="2023-05-15"
)


def generate_answer(conversation):
    response = client.chat.completions.create(
        model="gpt",
        messages=conversation,
        temperature=0.7,
        max_tokens=3000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )
    return (response.choices[0].message.content).strip()


# conversation = []

# Streamlit UI
st.title("Let's chat")

# Session state to store chat history
if "history" not in st.session_state:
    st.session_state["history"] = [
        {
            "role": "system",
            "content": "Am Anfang jeder Nachricht steht ein Kontext, den du verwendest, um deine Antworten zu erstellen. Dieser Kontext ist ein Auszug aus Versicherungstexten der DEVK. Du beantwortest hauptsächlich Fragen zu Versicherungsthemen mit Hilfe dieser Kontexte.",
        }
    ]

user_input = st.chat_input("You:", key="input")

if user_input:
    # get matching text from Azure AI Search and create prompt
    context = "\n".join(get_doc_azure_ai(user_input))
    st.session_state.history.append(
        {"role": "user", "content": f"Context: {context}\n\n{user_input}"}
    )
    rag_messages = f"context: {context}"

    # generate response and append it
    response = generate_answer(st.session_state.history)
    st.session_state.history.append({"role": "assistant", "content": response})

# Display chat history
for message in st.session_state.history:
    if message["role"] == "user":
        output = message["content"].split("\n\n", 1)
        # Using st.write for better formatting with Markdown
        st.write(f"**You:** {output[1]}")
    if message["role"] == "assistant":
        # Using st.write for better formatting with Markdown
        st.write(f"**ChatGPT:** {message['content']}")
