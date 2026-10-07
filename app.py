import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

st.set_page_config(
    page_title="Research RAG",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Research RAG")
st.write("Upload a research paper PDF and ask questions about it.")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


def extract_text(pdf_file):
    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def split_text(text, chunk_size=500):
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])

        if chunk.strip():
            chunks.append(chunk)

    return chunks


def create_database(chunks):
    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name="research_documents"
    )

    embeddings = embedding_model.encode(
        chunks
    ).tolist()

    ids = []

    for i in range(len(chunks)):
        ids.append("chunk_" + str(i))

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )

    return collection


def retrieve_information(collection, question):
    question_embedding = embedding_model.encode(
        [question]
    ).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )

    if results["documents"]:
        return results["documents"][0]

    return []


uploaded_file = st.file_uploader(
    "Upload your research paper",
    type=["pdf"]
)


if uploaded_file is not None:

    with st.spinner("Reading the research paper..."):
        text = extract_text(uploaded_file)

    if not text.strip():

        st.error(
            "No text could be extracted from this PDF."
        )

    else:

        st.success("Research paper uploaded successfully!")

        st.write(
            "Total characters extracted:",
            len(text)
        )

        chunks = split_text(text)

        st.write(
            "Number of text chunks:",
            len(chunks)
        )

        with st.spinner(
            "Creating the RAG database..."
        ):
            collection = create_database(chunks)

        st.success(
            "Research paper is ready for questions!"
        )

        st.subheader("🔎 Ask a Question")

        question = st.text_input(
            "Enter your question:"
        )

        if question:

            with st.spinner(
                "Searching the research paper..."
            ):

                relevant_documents = retrieve_information(
                    collection,
                    question
                )

            st.subheader("📖 Retrieved Information")

            if relevant_documents:

                for i, document in enumerate(
                    relevant_documents
                ):

                    st.markdown(
                        "### Source " + str(i + 1)
                    )

                    st.write(document)

            else:

                st.warning(
                    "No relevant information was found."
                )
else:

    st.info(
        "Please upload a research paper PDF to begin."
    )