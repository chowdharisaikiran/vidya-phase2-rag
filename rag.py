
import os
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

from conversation import contextualize_question


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
# LOAD DOCUMENTS RECURSIVELY
# =====================================

def load_documents():

    documents = []

    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for file in DOCUMENTS_DIR.rglob("*"):

        if not file.is_file():
            continue

        if file.suffix.lower() == ".pdf":

            try:
                documents.extend(
                    PyPDFLoader(str(file)).load()
                )
                print(f"Loaded PDF: {file.name}")

            except Exception as error:
                print(f"PDF loading error ({file.name}): {error}")

        elif file.suffix.lower() == ".txt":

            try:
                documents.extend(
                    TextLoader(
                        str(file),
                        encoding="utf-8"
                    ).load()
                )
                print(f"Loaded TXT: {file.name}")

            except Exception as error:
                print(f"TXT loading error ({file.name}): {error}")

    return documents


# =====================================
# CREATE FAISS VECTOR STORE
# =====================================

def create_vector_store():

    print("\nLoading educational documents...")

    docs = load_documents()

    if not docs:
        raise ValueError(
            "No PDF or TXT documents found in documents folder."
        )

    print(
        f"\nTotal document pages/files loaded: {len(docs)}"
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

    print(
        f"Generating embeddings for {len(chunks)} chunks..."
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

    print(
        "\nFAISS knowledge base created successfully!"
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
# SMART DOCUMENT RETRIEVAL
# =====================================

def retrieve_relevant_documents(question: str):

    index_file = VECTOR_DIR / "index.faiss"

    if not index_file.exists():
        return "", []

    vector_store = get_vector_store()

    results = vector_store.similarity_search_with_score(
        question.strip(),
        k=6
    )

    if not results:
        return "", []

    source_groups = defaultdict(list)

    for doc, score in results:

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        source_groups[source].append(
            (doc, score)
        )

    primary_source = min(
        source_groups,
        key=lambda source: (
            -len(source_groups[source]),
            min(score for _, score in source_groups[source])
        )
    )

    selected_results = sorted(
        source_groups[primary_source],
        key=lambda item: item[1]
    )[:3]

    relevant_docs = [
        doc for doc, score in selected_results
    ]

    context = "\n\n".join(
        doc.page_content
        for doc in relevant_docs
    )

    source_files = [
        Path(primary_source).name
    ]

    print("\nRetrieval Debug:")
    print("Selected Source:", source_files[0])
    print("Selected Chunks:", len(relevant_docs))

    return context, source_files


# =====================================
# TEXT ENCODING REPAIR
# =====================================

def repair_text_encoding(text: str) -> str:

    if not isinstance(text, str):
        text = str(text)

    if "Ã" in text or "â" in text:

        try:
            repaired = text.encode("cp1252").decode("utf-8")

            if repaired.count(" ") <= text.count(" "):
                return repaired

        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

    return text


# =====================================
# VIDYA AI - CONVERSATIONAL RAG
# =====================================

def ask_vidya(question: str, chat_history=None):

    if not question or not question.strip():
        raise ValueError(
            "Please enter a valid question."
        )

    chat_history = chat_history or []

    # Initialize Groq LLM
    llm = ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0.3
    )

    # =================================
    # CONVERSATIONAL QUESTION REWRITING
    # =================================

    standalone_question = contextualize_question(
        llm,
        question.strip(),
        chat_history
    )

    print("\nOriginal Question:", question)
    print("Contextualized Question:", standalone_question)

    # =================================
    # DOCUMENT RETRIEVAL
    # =================================

    context, source_files = retrieve_relevant_documents(
        standalone_question
    )

    # =================================
    # VIDYA PROMPT
    # =================================

    prompt = ChatPromptTemplate.from_template("""
You are Vidya, an intelligent AI learning assistant.

Your purpose is to help students learn effectively.

Instructions:

1. When educational context is provided,
   use it as the primary reference for answering.

2. Use only information from the context
   when making claims about uploaded documents.

3. If the context does not contain enough information,
   supplement the answer using your general knowledge.

4. Never refuse a general academic question
   just because it is missing from the documents.

5. Explain concepts in simple and clear language.

6. Provide examples and step-by-step explanations
   whenever useful.

7. For programming questions, explain concepts
   and provide code examples when appropriate.

8. Do not invent facts. If uncertain, say so clearly.

9. Respond in the same language used by the student.

10. Use standard ASCII punctuation wherever possible.

11. Maintain proper spaces between all words.

12. Use clean and simple Markdown formatting.

13. Avoid unnecessary tables and complicated formatting.

14. Keep answers accurate, structured, and readable.

15. Check punctuation and spacing before responding.

Educational context:
{context}

Conversation history:
{history}

Student's original question:
{original_question}

Contextualized question:
{question}

Answer:
""")

    # Format recent history for the answer prompt
    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in chat_history[-10:]
    )

    # =================================
    # GENERATE ANSWER
    # =================================

    chain = prompt | llm

    response = chain.invoke({
        "context": (
            context if context
            else "No relevant educational documents found."
        ),
        "history": history_text or "No previous conversation.",
        "original_question": question.strip(),
        "question": standalone_question
    })

    answer = repair_text_encoding(response.content)

    # =================================
    # SOURCE ATTRIBUTION
    # =================================

    if source_files:

        return (
            f"{answer}\n\n"
            f"**Knowledge Base Sources:** "
            f"{', '.join(source_files)}"
        )

    return (
        f"{answer}\n\n"
        "**Source:** General AI (Groq)"
    )


# =====================================
# LOCAL CONVERSATION TEST
# =====================================

if __name__ == "__main__":

    print("\nVIDYA AI ASSISTANT")
    print("-" * 40)

    chat_history = []

    while True:

        question = input(
            "\nAsk Vidya (type exit to quit): "
        ).strip()

        if question.lower() in ["exit", "quit"]:

            print("Thank you for using Vidya!")
            break

        try:

            answer = ask_vidya(
                question,
                chat_history
            )

            print("\nVidya:")
            print(answer)

            chat_history.append({
                "role": "user",
                "content": question
            })

            chat_history.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as error:

            print("\nError:", error)
