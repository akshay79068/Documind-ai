from langchain_huggingface import HuggingFaceEmbeddings


def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


if __name__ == "__main__":
    embeddings = create_embeddings()

    text = "Artificial Intelligence and Machine Learning"

    vector = embeddings.embed_query(text)

    print("Vector length:", len(vector))
    print("First 10 values:", vector[:10])
