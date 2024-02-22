# RAG-Workshop
This repo contains an example of retrieval augmented generation (RAG) using ChatGPT, Azure AI Search and Streamlit

## Prerequisites

1. Create an Azure OpenAI Resource
2. Create an Azure AI Search Resource
3. Create `.env`-file in scripts directory and add the following environment variables:
    ```python 
    AZURE_OPENAI_KEY ="your AzureOpenAI key" 
    AZURE_OPENAI_ENDPOINT ="your AzureOpenAI endpoint"
    EMBEDDING_DEPLOYMENT_NAME = "the name of your deployed text-embedding model"
    CHAT_DEPLOYMENT_NAME = "gpt-35-turbo"

    AZURE_SEARCH_SERVICE_ENDPOINT='your Azure AI Search Service endpoint'
    AZURE_SEARCH_INDEX_NAME = 'your indexname'
    AZURE_SEARCH_ADMIN_KEY = 'your Azure AI Search Service admin key'
    ```
4. Setup virtual python environment by running the following commands in the root directory of your project:
    ```
    python -m venv venv
    ./venv/Scripts/activate
    pip install -r .\requirements.txt
    ```

## How to run the files
- Always execute each file from within its folder if it is a plain python script
- Watch out to have the virtual python enviroment activated by once running `./venv/Scripts/activate`
