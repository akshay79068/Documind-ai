from langchain_community.document_loaders import PyPDFLoader
from chunking import split_documents


def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


if __name__ == "__main__":
    documents = load_pdf(
        "C:/Users/acer/Downloads/AKSHAYSHARMA (4).pdf"
    )

    chunks = split_documents(documents)

    print("Total pages:", len(documents))
    print("Total chunks:", len(chunks))

    print("\nFirst chunk:")
    print(chunks[0].page_content)