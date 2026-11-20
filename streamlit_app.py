
import os
import requests
import streamlit as st

from dotenv import load_dotenv
from supabase import create_client

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

API_URL = os.getenv(
    "VIDYA_API_URL",
    "http://127.0.0.1:8000"
)

st.set_page_config(
    page_title="Vidya AI Assistant",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

defaults = {
    "logged_in": False,
    "access_token": None,
    "email": "",
    "session_id": None,
    "messages": []
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# LOGIN FUNCTION
# --------------------------------------------------

def login_user(email, password):

    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        st.error("Supabase configuration missing in .env")
        return

    try:
        supabase = create_client(
            SUPABASE_URL,
            SUPABASE_ANON_KEY
        )

        result = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })

        if not result.session:
            st.error("Login failed. Check your credentials.")
            return

        st.session_state.logged_in = True
        st.session_state.access_token = (
            result.session.access_token
        )
        st.session_state.email = email
        st.session_state.session_id = None
        st.session_state.messages = []

        st.rerun()

    except Exception as error:
        st.error(f"Login failed: {error}")


# --------------------------------------------------
# ASK VIDYA API
# --------------------------------------------------

def ask_vidya(question):

    headers = {
        "Authorization": (
            f"Bearer {st.session_state.access_token}"
        ),
        "Content-Type": "application/json"
    }

    payload = {
        "question": question
    }

    if st.session_state.session_id:
        payload["session_id"] = st.session_state.session_id

    try:
        response = requests.post(
            f"{API_URL}/ask",
            headers=headers,
            json=payload,
            timeout=180
        )

        if response.status_code == 200:

            data = response.json()

            st.session_state.session_id = (
                data.get("session_id")
            )

            return data

        if response.status_code == 401:
            st.error("Session expired. Please log in again.")
            return None

        st.error(
            f"API Error {response.status_code}: "
            f"{response.text}"
        )

        return None

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to Vidya API. "
            "Please start the FastAPI backend."
        )
        return None

    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return None

    except Exception as error:
        st.error(f"Unexpected error: {error}")
        return None


# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------

if not st.session_state.logged_in:

    st.title("🎓 Vidya AI Assistant")

    st.caption(
        "Your personalized educational AI assistant"
    )

    st.divider()

    st.subheader("Login")

    with st.form("login_form"):

        email = st.text_input("Email address")

        password = st.text_input(
            "Password",
            type="password"
        )

        submitted = st.form_submit_button(
            "Login",
            use_container_width=True
        )

        if submitted:

            if not email or not password:
                st.warning("Enter email and password.")

            else:
                login_user(email, password)

    st.stop()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🎓 Vidya")

    st.write("Logged in as:")

    st.caption(st.session_state.email)

    st.divider()

    if st.button("New Chat", use_container_width=True):

        st.session_state.session_id = None
        st.session_state.messages = []

        st.rerun()

    if st.button("Logout", use_container_width=True):

        st.session_state.logged_in = False
        st.session_state.access_token = None
        st.session_state.email = ""
        st.session_state.session_id = None
        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.caption("Powered by FastAPI, Supabase and RAG")


# --------------------------------------------------
# CHAT INTERFACE
# --------------------------------------------------

st.title("Vidya AI Assistant")

st.caption(
    "Ask questions and learn with your AI assistant."
)

st.divider()


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("Knowledge Base Sources"):

                for source in message["sources"]:
                    st.write(f"📄 {source}")


# Chat input

question = st.chat_input("Ask Vidya anything...")


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Vidya is thinking..."):

            data = ask_vidya(question)

        if data:

            answer = data.get(
                "answer",
                "No answer received."
            )

            sources = data.get("sources", [])

            st.markdown(answer)

            if sources:

                with st.expander("Knowledge Base Sources"):

                    for source in sources:
                        st.write(f"📄 {source}")

            source_type = data.get(
                "source_type",
                "general_ai"
            )

            if source_type == "knowledge_base":

                st.success("Knowledge Base Used")

            else:

                st.info("General AI Response")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        else:

            st.warning(
                "Vidya could not process this question."
            )
