
import os 
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from get_embedding_function import get_embedding_function

CHROMA_PATH = "./Chroma-Constitucion"

# ======================================================================
# Functions
# ======================================================================
def calculate_chunk_ids(chunks):

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

# ======================================================================
# Code 
# ======================================================================

# Load the document and split it into chunks
loader = TextLoader("./Documents/Constitucion.txt")
documents = loader.load()

# Split it into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=10,
    length_function=len,
)
chunks = text_splitter.split_documents(documents)



db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

# Calculate Page IDs.
chunks_with_ids = calculate_chunk_ids(chunks)

# Add or Update the documents.
existing_items = db.get(include=[])  # IDs are always included by default
existing_ids = set(existing_items["ids"])
print(f"Number of existing documents in DB: {len(existing_ids)}")

# Only add documents that don't exist in the DB.
new_chunks = []
for chunk in chunks_with_ids:
    if chunk.metadata["id"] not in existing_ids:
        new_chunks.append(chunk)

if len(new_chunks):
    print(f"👉 Adding new documents: {len(new_chunks)}")
    new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
    db.add_documents(new_chunks, ids=new_chunk_ids)
    db.persist()
else:
    print("✅ No new documents to add")

print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
