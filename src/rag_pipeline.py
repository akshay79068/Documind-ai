from vectorstore import (
    create_vector_store,
    search_documents
)

from embeddings import create_embeddings

import os
from dotenv import load_dotenv
from groq import Groq


# Load environment variables from .env
load_dotenv()


# Create Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def ask_question(question):
    """
    Retrieve relevant document chunks and
    generate an answer using the LLM.
    """

    # 1. Create embedding model
    embeddings = create_embeddings()

    # 2. Open ChromaDB collection
    collection = create_vector_store()

    # 3. Retrieve the most relevant chunks
    results = search_documents(
        collection,
        question,
        embeddings,
        k=3
    )

    # 4. Prepare context with source page information
    context_parts = []

    for i, document in enumerate(results["documents"][0]):

        metadata = results["metadatas"][0][i]

        page = metadata.get(
            "page_label",
            "Unknown"
        )

        context_parts.append(
            f"[Page {page}]\n{document}"
        )

    context = "\n\n".join(context_parts)

    # 5. Create the RAG prompt
    prompt = f"""
You are a helpful document assistant.

Use only the information provided in the context
to answer the user's question.

If the answer is not available in the context,
say that the information is not available in the document.

Context:
{context}

Question:
{question}

Answer:
"""

    # 6. Send prompt to the LLM
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # 7. Extract the generated answer
    answer = response.choices[0].message.content

    # 8. Extract source pages
    source_pages = sorted(
        set(
            metadata.get(
                "page_label",
                "Unknown"
            )
            for metadata in results["metadatas"][0]
        )
    )

    # 9. Return answer + source pages
    return answer, source_pages


# Test the RAG pipeline
if __name__ == "__main__":

    question = "What programming skills does the candidate have?"

    answer, source_pages = ask_question(question)

    print("\nAnswer:")
    print(answer)

    print("\nSources:")

    for page in source_pages:
        print(f"- Page {page}")