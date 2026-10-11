import uuid
import chromadb

from pdf_loader import load_pdf
from chunking import split_documents
from embeddings import create_embeddings


def create_vector_store():
    """
    Create or open a persistent ChromaDB database
    and return the DocuMind collection.
    """

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="documind"
    )

    return collection


def add_documents(collection, chunks, embeddings):
    """
    Convert each document chunk into an embedding
    and store the text, vector, and metadata
    in ChromaDB.
    """

    for i, chunk in enumerate(chunks):

        vector = embeddings.embed_query(
            chunk.page_content
        )

        collection.add(
            ids=[f"{uuid.uuid4()}_{i}"],
            documents=[chunk.page_content],
            embeddings=[vector],
            metadatas=[chunk.metadata]
        )


def process_pdf(file_path, collection, embeddings):
    """
    Load a PDF, split it into chunks,
    create embeddings, and store them in ChromaDB.
    """

    # --------------------------------------------------
    # 1. Clear previous document
    # --------------------------------------------------

    existing_data = collection.get()

    existing_ids = existing_data["ids"]

    if existing_ids:
        collection.delete(
            ids=existing_ids
        )

    # --------------------------------------------------
    # 2. Load PDF
    # --------------------------------------------------

    documents = load_pdf(file_path)

    # --------------------------------------------------
    # 3. Split PDF into chunks
    # --------------------------------------------------

    chunks = split_documents(documents)

    print(
        "Total chunks created:",
        len(chunks)
    )

    # --------------------------------------------------
    # 4. Store chunks in ChromaDB
    # --------------------------------------------------

    add_documents(
        collection,
        chunks,
        embeddings
    )

    # --------------------------------------------------
    # 5. Return chunk count
    # --------------------------------------------------

    return len(chunks)


def search_documents(collection, query, embeddings, k=3):
    """
    Convert the user's query into an embedding
    and retrieve the top-k most relevant chunks.
    """

    query_vector = embeddings.embed_query(
        query
    )

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k
    )

    return results


if __name__ == "__main__":

    # --------------------------------------------------
    # 1. Create embedding model
    # --------------------------------------------------

    embeddings = create_embeddings()

    # --------------------------------------------------
    # 2. Create / open ChromaDB collection
    # --------------------------------------------------

    collection = create_vector_store()

    # --------------------------------------------------
    # 3. Process PDF
    # --------------------------------------------------

    total_chunks = process_pdf(
        "C:/Users/acer/Downloads/AKSHAYSHARMA (4).pdf",
        collection,
        embeddings
    )

    print(
        "Total chunks stored:",
        collection.count()
    )

    # --------------------------------------------------
    # 4. Test similarity search
    # --------------------------------------------------

    query = "What programming skills does the candidate have?"

    results = search_documents(
        collection,
        query,
        embeddings,
        k=3
    )

    # --------------------------------------------------
    # 5. Display retrieved chunks
    # --------------------------------------------------

    print("\nRelevant chunks:\n")

    for i, document in enumerate(
        results["documents"][0]
    ):

        print(
            f"--- Result {i + 1} ---"
        )

        print(document)

        print("Metadata:")

        print(
            results["metadatas"][0][i]
        )

        print()
    