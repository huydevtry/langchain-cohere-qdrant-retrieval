from flask import Flask, request
from flask_cors import CORS
import json
import requests
from bs4 import BeautifulSoup
import re
import os
import pprint

# Loading environment variables
import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.environ.get('openai_api_key')
cohere_api_key = os.environ.get('cohere_api_key')
qdrant_url = os.environ.get('qdrant_url')
qdrant_api_key = os.environ.get('qdrant_api_key')

#Flask config
app = Flask(__name__)
CORS(app)

# Test default route
@app.route('/')
def hello_world():
    return {"Hello":"World"}

## Embedding code
from langchain.embeddings import CohereEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Qdrant


@app.route('/embed', methods=['POST'])
def embed_pdf():
    collection_name = request.json.get("collection_name")
    file_url = request.json.get("file_url")

    loader = PyPDFLoader(file_url)
    docs = loader.load_and_split()
    embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
    qdrant = Qdrant.from_documents(docs, embeddings, url=qdrant_url, collection_name=collection_name, prefer_grpc=True, api_key=qdrant_api_key)
    
    return {"collection_name":qdrant.collection_name}

# Retrieve information from a collection
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from qdrant_client import QdrantClient

@app.route('/retrieve', methods=['POST'])
def retrieve_info():
    collection_name = request.json.get("collection_name")
    query = request.json.get("query")

    client = QdrantClient(url=qdrant_url, prefer_grpc=True, api_key=qdrant_api_key)

    embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
    qdrant = Qdrant(client=client, collection_name=collection_name, embedding_function=embeddings.embed_query)
    search_results = qdrant.similarity_search(query, k=2)
    chain = load_qa_chain(OpenAI(openai_api_key=openai_api_key,temperature=0.2), chain_type="stuff")
    results = chain({"input_documents": search_results, "question": query}, return_only_outputs=True)
    
    return {"results":results["output_text"]}

@app.route('/scrape', methods=['POST'])
def scrape_site():
    url = request.json.get("url")

    response = requests.get(url)

    soup = BeautifulSoup(response.content, 'html.parser')
    links = soup.find_all('p')
    context = ''
    for link in links:
        context += link.text + " "
    clean_context = context.replace('\n', '').replace('\r', '').strip()

    return {"content":clean_context}

from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

@app.route('/serper', methods=['POST'])
def google_serper():
    # os.environ['OPENAI_API_KEY'] = "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"
    # os.environ["SERPER_API_KEY"] = "a9df4278dd394022aa9be10785eb6fd30825615d"

    query = request.json.get("query")
    llm = OpenAI(temperature=0)
    search = GoogleSerperAPIWrapper()
    tools = [
        Tool(
            name="Intermediate Answer",
            func=search.run,
            description="useful for when you need to ask with search"
        )
    ]

    self_ask_with_search = initialize_agent(tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)
    response = self_ask_with_search.run(query)
    

    return {"content":response}