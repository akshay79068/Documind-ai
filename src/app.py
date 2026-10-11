import os
import tempfile

import streamlit as st

from embeddings import create_embeddings
from vectorstore import (
    create_vector_store,
    process_pdf
)
from rag_pipeline import ask_question

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="centered"
)


st.title("📄 DocuMind AI")

st.write(
    "Upload a PDF and ask questions about its content."
)


uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# --------------------------------------------------
# Process PDF
# --------------------------------------------------

if uploaded_file is not None:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    if st.button("Process Document"):

        
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(
                uploaded_file.getvalue()
            )

            temp_path = temp_file.name

        try:

            with st.spinner(
                "Processing document..."
            ):

                # Create embedding model
                embeddings = create_embeddings()

                # Open ChromaDB
                collection = create_vector_store()

                # Process PDF
                total_chunks = process_pdf(
                    temp_path,
                    collection,
                    embeddings
                )

            # Save processing status
            st.session_state["document_processed"] = True

            st.session_state["file_name"] = (
                uploaded_file.name
            )

            st.success(
                f"Document processed successfully! "
                f"{total_chunks} chunks created."
            )

        finally:

            
            if os.path.exists(temp_path):
                os.remove(temp_path)




if st.session_state.get(
    "document_processed",
    False
):

    st.divider()

    st.subheader(
        "💬 Ask a Question"
    )

    question = st.text_input(
        "Ask something about your document"
    )

    if st.button("Ask Question"):

        if question.strip() == "":

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching the document and generating answer..."
            ):

                answer, source_pages = (
                    ask_question(question)
                )


            st.subheader("Answer")

            st.write(answer)

            

            st.subheader("Sources")

            if source_pages:

                for page in source_pages:

                    st.write(
                        f"📄 Page {page}"
                    )

            else:
#my name is akshay sharma aaa
                st.write(
                    "No source page found."
                )