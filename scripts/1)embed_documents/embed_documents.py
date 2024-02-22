from dotenv import load_dotenv
import os
from openai import AzureOpenAI
import re
import json
import glob
from PyPDF2 import PdfReader

# Load environment variables from a .env file
load_dotenv()

# Retrieve API key and endpoint from environment variables
api_key = os.getenv("AZURE_OPENAI_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# Specify the API version and the engine to be used for embeddings
api_version = "2023-12-01-preview"
engine = os.getenv("EMBEDDING_DEPLOYMENT_NAME")

# Initialize the AzureOpenAI client
client = AzureOpenAI(
    api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint
)


def get_files_from_data_dir(directory_path="../../data"):
    """
    Retrieves all PDF file paths from the specified directory.

    Parameters:
        directory_path (str): Relative path to the directory containing PDF files.

    Returns:
        list: A list of absolute file paths for each PDF file found.
    """
    # Convert the relative directory path to an absolute path
    absolute_path = os.path.abspath(directory_path)
    print(f"Scanning for PDF files in: {absolute_path}")

    # List all PDF files in the directory
    pdf_files = glob.glob(f"{absolute_path}/*.pdf")
    print(f"Found {len(pdf_files)} PDF file(s) in the directory.")

    return pdf_files


def get_pdf_text(file_path):
    """
    Extracts text from a PDF file.

    Parameters:
        file_path (str): The absolute path to the PDF file.

    Returns:
        str: The extracted text from the PDF file.
    """
    try:
        reader = PdfReader(file_path)
        full_text = ""

        # Iterate over each page in the PDF file and extract text
        for page in reader.pages:
            full_text += (
                page.extract_text() or ""
            )  # Append text or an empty string if None

        print(f"Successfully extracted text from: {file_path}")
        return full_text
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def normalize_text(text):
    """
    Normalizes the given text by removing unnecessary characters and spaces.

    Parameters:
        text (str): The text to normalize.

    Returns:
        str: The normalized text.
    """
    text = re.sub(
        r"\s+", " ", text
    ).strip()  # Replace multiple spaces with a single space
    text = re.sub(r"[. ]+,", "", text)  # Remove spaces before commas
    text = text.replace("..", ".").replace(". .", ".").replace("\n", "").strip()
    return text


def get_chunks(text, chunk_length=500):
    """
    Divides the text into chunks of specified length, preferably breaking at sentence endings.

    Parameters:
        text (str): The text to divide into chunks.
        chunk_length (int): The maximum length of each chunk.

    Returns:
        list: A list of text chunks.
    """
    chunks = []
    while len(text) > chunk_length:
        # Find the last period in the chunk to avoid breaking a sentence
        last_period_index = text[:chunk_length].rfind(".")
        if last_period_index == -1:  # If no period is found, use the chunk length
            last_period_index = chunk_length

        # Append the chunk and remove it from the text
        chunks.append(text[: last_period_index + 1])
        text = text[last_period_index + 1 :]

    # Add any remaining text as a chunk
    if text:
        chunks.append(text)

    return chunks


def get_embedding(engine, text_chunks, filename):
    """
    Generates embeddings for each chunk of text using Azure OpenAI and assigns them to a document.

    Parameters:
        engine (str): The engine to use for generating embeddings.
        text_chunks (list): A list of text chunks for which to generate embeddings.
        filename (str): The name of the file associated with the text chunks.

    Returns:
        list: A list of dictionaries containing embeddings and metadata for each text chunk.
    """
    document_embeddings = []
    for counter, chunk in enumerate(text_chunks):
        try:
            response = client.embeddings.create(input=chunk, model=engine)
            embedding_data = {
                "id": str(counter),
                "line": chunk,
                "embedding": response.data[0].embedding,
                "filename": filename,
            }
            document_embeddings.append(embedding_data)
            print(f"Generated embedding for chunk {counter} of {filename}")
        except Exception as e:
            print(f"Error generating embedding for chunk {counter} of {filename}: {e}")
    return document_embeddings


def ensure_output_directory_exists(directory_path):
    """
    Checks if the specified directory exists, and if not, creates it.

    Parameters:
        directory_path (str): The path to the directory to check and potentially create.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")


# Main process
if __name__ == "__main__":
    # Define the output directory path
    output_directory = "../output"
    # Ensure the output directory exists
    ensure_output_directory_exists(output_directory)

    # Get PDF files from the data directory
    pdf_files = get_files_from_data_dir()

    for file_path in pdf_files:
        # Extract, normalize, and chunk text from each PDF file
        raw_text = get_pdf_text(file_path)
        normalized_text = normalize_text(raw_text)
        text_chunks = get_chunks(normalized_text)

        # Generate embeddings for the text chunks
        embeddings = get_embedding(engine, text_chunks, os.path.basename(file_path))

        # Write the embeddings to a JSON file in the output directory
        output_file = f"{output_directory}/{os.path.basename(file_path)}.json"
        with open(output_file, "w+", encoding="utf-8") as outfile:
            json.dump(embeddings, outfile)
        print(f"Embeddings written to {output_file}")
