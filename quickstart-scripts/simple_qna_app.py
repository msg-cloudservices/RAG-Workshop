import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMChain, ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import  ChatMessageHistory, ConversationBufferWindowMemory

import streamlit as st

load_dotenv()

#write original deployment name here - since your deployment-name has to be the same as the original, you can use CHAT_DEPLOYMENT_NAME here
deployment_name=os.getenv("CHAT_DEPLOYMENT_NAME")

#importing Azure OpenAI creds
api_key = os.getenv("AZURE_OPENAI_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
model=os.getenv("CHAT_DEPLOYMENT_NAME")

#create instance of AzureOpenAI
llm = AzureChatOpenAI(
    api_version="2023-12-01-preview",
    api_key=api_key,  
    azure_endpoint = azure_endpoint,
    deployment_name=deployment_name, 
    model_name=model,
    temperature=0.9,
    )

#create frontend
st.title("🦜️🔗Langchain LLM Demo App")
prompt = st.text_input("Enter your question here")

memory = ConversationBufferWindowMemory(k=10)

conversation = ConversationChain(llm=llm, verbose=True, memory=memory)

if prompt:
   response = conversation.predict(input=prompt)
   
   st.write(response)


# to run this script, open a terminal and type: streamlit run .\quickstart-scripts\simple_qna_app.py

# if you want to stop the script, run Ctrl C 