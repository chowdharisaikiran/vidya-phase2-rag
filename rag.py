
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

DOCUMENTS_DIR = Path("documents")
VECTOR_DIR = Path("embeddings")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def load_documents():
    documents = []

    for file in DOCUMENTS_DIR.iterdir():
        if file.suffix.lower() == ".pdf":
            documents.extend(PyPDFLoader(str(file)).load())
        elif file.suffix.lower() == ".txt":
            documents.extend(TextLoader(str(file), encoding="utf-8").load())

    if not documents:
        raise ValueError("No PDF or TXT documents found in documents folder.")

    return documents


def create_vector_store():
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(VECTOR_DIR))

    return len(chunks)


def get_vector_store():
    if not (VECTOR_DIR / "index.faiss").exists():
        raise FileNotFoundError("Vector database not found. Create it first.")

    return FAISS.load_local(
        str(VECTOR_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )


def ask_vidya(question: str):
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is missing in .env")

    vector_store = get_vector_store()
    relevant_docs = vector_store.similarity_search(question, k=4)

    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
You are Vidya, an educational AI assistant.
Answer the student's question using the provided context.
If the answer is not available in the context, clearly say so.
Do not invent academic facts.

Context:
{context}

Student question:
{question}
""")

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content
