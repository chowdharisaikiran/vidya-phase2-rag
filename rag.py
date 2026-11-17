
import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq


# =====================================
# CONFIGURATION
# =====================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

DOCUMENTS_DIR = BASE_DIR / "documents"
VECTOR_DIR = BASE_DIR / "embeddings"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing in .env")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing in .env")


# =====================================
# GEMINI EMBEDDINGS
# =====================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)


# =====================================
# LOAD DOCUMENTS
# =====================================

def load_documents():

    documents = []

    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for file in DOCUMENTS_DIR.iterdir():

        if file.suffix.lower() == ".pdf":

            documents.extend(
                PyPDFLoader(str(file)).load()
            )

        elif file.suffix.lower() == ".txt":

            documents.extend(
                TextLoader(
                    str(file),
                    encoding="utf-8"
                ).load()
            )

    return documents


# =====================================
# CREATE FAISS VECTOR STORE
# =====================================

def create_vector_store():

    docs = load_documents()

    if not docs:
        raise ValueError(
            "No PDF or TXT documents found in documents folder."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError(
            "No document chunks were generated."
        )

    VECTOR_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    vector_store.save_local(
        str(VECTOR_DIR)
    )

    return len(chunks)


# =====================================
# LOAD FAISS VECTOR STORE
# =====================================

def get_vector_store():

    index_file = VECTOR_DIR / "index.faiss"

    if not index_file.exists():
        raise FileNotFoundError(
            "FAISS index not found. "
            "Create the vector store first."
        )

    return FAISS.load_local(
        str(VECTOR_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )


# =====================================
# VIDYA AI - RAG + GENERAL AI
# =====================================

def ask_vidya(question: str):

    if not question or not question.strip():
        raise ValueError(
            "Please enter a valid question."
        )

    # Initialize Groq LLM
    llm = ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0.3
    )

    context = ""

    # Retrieve documents only when FAISS index exists
    if (VECTOR_DIR / "index.faiss").exists():

        vector_store = get_vector_store()

        results = vector_store.similarity_search_with_score(
            question,
            k=4
        )

        # Lower FAISS L2 distance means better similarity.
        # Threshold is a configurable starting point.
        relevant_docs = [
            doc
            for doc, score in results
            if score < 1.2
        ]

        context = "\n\n".join(
            doc.page_content
            for doc in relevant_docs
        )

    # Prompt for both RAG and general knowledge
    prompt = ChatPromptTemplate.from_template("""
You are Vidya, an intelligent AI learning assistant.

Your purpose is to help students learn effectively.

Instructions:

1. If relevant educational context is provided,
   use it to answer the student's question.

2. If the context does not contain the answer,
   answer using your general knowledge.

3. Never refuse to answer a general academic question
   just because it is missing from the documents.

4. Explain concepts in simple language.

5. Provide examples and step-by-step explanations
   whenever useful.

6. For programming questions, explain concepts and
   provide code examples when appropriate.

7. Do not invent facts. If uncertain, say so clearly.

8. Respond in the same language used by the student.

Educational context:
{context}

Student question:
{question}

Answer:
""")

    chain = prompt | llm

    response = chain.invoke({
        "context": context if context else
        "No relevant educational documents found.",
        "question": question.strip()
    })

    return response.content


# =====================================
# LOCAL TEST
# =====================================

if __name__ == "__main__":

    print("\n🎓 VIDYA AI ASSISTANT")
    print("-" * 40)

    while True:

        question = input(
            "\nAsk Vidya (type exit to quit): "
        ).strip()

        if question.lower() in ["exit", "quit"]:
            print("Thank you for using Vidya!")
            break

        try:

            answer = ask_vidya(question)

            print("\nVidya:")
            print(answer)

        except Exception as error:

            print("\nError:", error)
