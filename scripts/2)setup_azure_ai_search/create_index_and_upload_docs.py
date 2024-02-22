# Import necessary libraries
from dotenv import load_dotenv
import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    HnswParameters,
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
)
import glob
import json

# Load environment variables
load_dotenv()
print("Environment variables loaded.")

# Retrieve necessary configuration values from environment variables
service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
key = os.getenv("AZURE_SEARCH_ADMIN_KEY")

# Validate that necessary configurations are not None
if not all([service_endpoint, index_name, key]):
    raise ValueError(
        "One or more environment variables (endpoint, index name, key) are missing."
    )

# Set up the credential for Azure services
credential = AzureKeyCredential(key)

# Initialize the SearchIndexClient
index_client = SearchIndexClient(endpoint=service_endpoint, credential=credential)
print(f"SearchIndexClient initialized for endpoint: {service_endpoint}")

# Define the schema of the index with fields
fields = [
    # Define a simple field that acts as a unique identifier for each document
    SimpleField(
        name="id",
        type=SearchFieldDataType.String,
        key=True,
        sortable=True,
        filterable=True,
        facetable=True,
    ),
    # Define fields that will be searchable
    SearchableField(name="line", type=SearchFieldDataType.String),
    SearchableField(
        name="filename",
        type=SearchFieldDataType.String,
        filterable=True,
        facetable=True,
    ),
    # Define a field to store vector embeddings for vector search
    SearchField(
        name="embedding",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=1536,
        vector_search_profile_name="myHnswProfile",
    ),
]

# Configure vector search settings
vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            name="myHnsw",
            kind=VectorSearchAlgorithmKind.HNSW,
            parameters=HnswParameters(
                m=4,
                ef_construction=400,
                ef_search=500,
                metric=VectorSearchAlgorithmMetric.COSINE,
            ),
        ),
        ExhaustiveKnnAlgorithmConfiguration(
            name="myExhaustiveKnn",
            kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
            parameters=ExhaustiveKnnParameters(
                metric=VectorSearchAlgorithmMetric.COSINE
            ),
        ),
    ],
    profiles=[
        VectorSearchProfile(
            name="myHnswProfile", algorithm_configuration_name="myHnsw"
        ),
        VectorSearchProfile(
            name="myExhaustiveKnnProfile",
            algorithm_configuration_name="myExhaustiveKnn",
        ),
    ],
)

# Set up semantic search configuration
semantic_config = SemanticConfiguration(
    name="my-semantic-config",
    prioritized_fields=SemanticPrioritizedFields(
        content_fields=[SemanticField(field_name="line")],
        keywords_fields=[SemanticField(field_name="filename")],
    ),
)

# Combine all configurations into the search index definition
index = SearchIndex(
    name=index_name,
    fields=fields,
    vector_search=vector_search,
    semantic_search=SemanticSearch(configurations=[semantic_config]),
)
# Create or update the index on Azure Search
result = index_client.create_or_update_index(index)
print(f"{result.name} created or updated successfully.")

# Preparing to upload documents to the index
cwd = os.getcwd()
relative_path = "../output"
absolute_path = os.path.abspath(os.path.join(cwd, relative_path))

# Load JSON documents from specified directory
files = []
for file in glob.glob("../output/*.json"):
    filepath = os.path.abspath(os.path.join(absolute_path, file))
    with open(filepath, "r") as infile:
        files.extend(json.load(infile))

# Initialize the SearchClient
search_client = SearchClient(
    endpoint=service_endpoint, index_name=index_name, credential=credential
)
# Upload documents to the index
result = search_client.upload_documents(documents=files)
print(f'Uploaded {len(files)} documents to the index "{index_name}".')
