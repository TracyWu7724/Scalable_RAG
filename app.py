import streamlit as st
import requests

API_BASE = "http://localhost:8000"

AVAILABLE_MODELS = [
    "BAAI_bge-base-en-v1-5",
    "intfloat_e5-base-v2",
    "nomic-ai_nomic-embed-text-v1",
    "sentence-transformers_all-MiniLM-L6-v2",
]

AVAILABLE_LLM_MODELS = [
    "gemini-2.5-flash",
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemma-3-4b-it",
]

st.set_page_config(page_title="Henkel Adhesive Chatbot", layout="centered")


# ── Helper: auth header ───────────────────────────────────────────
def _auth():
    return (st.session_state.get("username", ""), st.session_state.get("password", ""))


def _logged_in():
    return bool(st.session_state.get("logged_in"))


# ── Sidebar: Login / Register ─────────────────────────────────────
with st.sidebar:
    st.header("Account")

    if _logged_in():
        st.success(f"Logged in as **{st.session_state['username']}**")
        if st.button("Logout"):
            for key in ["logged_in", "username", "password", "conversation_id", "history", "conversations"]:
                st.session_state.pop(key, None)
            st.rerun()
    else:
        tab_login, tab_register = st.tabs(["Login", "Register"])

        with tab_login:
            login_user = st.text_input("Username", key="login_user")
            login_pass = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login"):
                # Validate credentials by calling a protected endpoint
                resp = requests.get(
                    f"{API_BASE}/conversations", auth=(login_user, login_pass)
                )
                if resp.status_code == 200:
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = login_user
                    st.session_state["password"] = login_pass
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        with tab_register:
            reg_user = st.text_input("Username", key="reg_user")
            reg_pass = st.text_input("Password", type="password", key="reg_pass")
            if st.button("Register"):
                resp = requests.post(
                    f"{API_BASE}/register",
                    json={"username": reg_user, "password": reg_pass},
                )
                if resp.status_code == 201:
                    st.success("Account created! You can now log in.")
                else:
                    st.error(resp.json().get("detail", "Registration failed"))


# ── Sidebar: Conversation list ─────────────────────────────────────

if _logged_in():
    with st.sidebar:
        st.divider()
        st.header("Conversations")

        if st.button("New Conversation"):
            st.session_state.pop("conversation_id", None)
            st.session_state["history"] = []
            st.rerun()

        # Fetch conversation list
        try:
            resp = requests.get(f"{API_BASE}/conversations", auth=_auth())
            if resp.status_code == 200:
                conversations = resp.json()
            else:
                conversations = []
        except Exception:
            conversations = []

        for conv in conversations:
            label = conv["title"][:40]
            if st.button(label, key=conv["id"]):
                # Load this conversation
                detail_resp = requests.get(
                    f"{API_BASE}/conversations/{conv['id']}", auth=_auth()
                )
                if detail_resp.status_code == 200:
                    detail = detail_resp.json()
                    st.session_state["conversation_id"] = conv["id"]
                    st.session_state["history"] = [
                        {"role": m["role"], "content": m["content"]}
                        for m in detail["messages"]
                    ]
                    st.rerun()


# ── Main chat area ─────────────────────────────────────────────────

st.title("Henkel RAG Chatbot")
st.markdown("Ask anything about Henkel adhesive products!")

col1, col2 = st.columns(2)
with col1:
    model_name = st.selectbox("Embedding Model", AVAILABLE_MODELS)
with col2:
    llm_model = st.selectbox("LLM Model", AVAILABLE_LLM_MODELS)

if "history" not in st.session_state:
    st.session_state.history = []

# Render chat history
for msg in st.session_state.history:
    role = msg["role"] if "role" in msg else ("user" if "user" in msg else "assistant")
    content = msg.get("content") or msg.get("user") or msg.get("bot", "")
    st.chat_message(role).write(content)

user_input = st.chat_input("Say something")

if user_input:
    # Show user message immediately
    st.chat_message("user").write(user_input)

    if not _logged_in():
        # Fallback: unauthenticated mode (no persistence)
        with st.spinner("Thinking..."):
            st.warning("Log in to save conversations.")
            # Can't call POST /ask without auth, so show a message
            st.session_state.history.append({"role": "user", "content": user_input})
            st.session_state.history.append(
                {"role": "assistant", "content": "Please log in to use the chatbot."}
            )
            st.rerun()
    else:
        with st.spinner("Thinking..."):
            payload = {
                "question": user_input,
                "model_name": model_name,
                "llm_model": llm_model,
                "conversation_id": st.session_state.get("conversation_id"),
            }

            try:
                resp = requests.post(
                    f"{API_BASE}/ask", json=payload, auth=_auth(), timeout=130
                )
                if resp.status_code != 200:
                    # Show the real server error
                    try:
                        detail = resp.json().get("detail", resp.text)
                    except Exception:
                        detail = resp.text or f"HTTP {resp.status_code}"
                    answer = f"API error ({resp.status_code}): {detail}"
                else:
                    data = resp.json()
                    answer = data.get("answer", "No answer received.")
                    st.session_state["conversation_id"] = data.get("conversation_id")
            except requests.ConnectionError:
                answer = "Cannot connect to API. Is the FastAPI server running on port 8000?"
            except requests.Timeout:
                answer = "Request timed out. The server took too long to respond."
            except Exception as e:
                answer = f"Error contacting API: {e}"

        st.session_state.history.append({"role": "user", "content": user_input})
        st.session_state.history.append({"role": "assistant", "content": answer})
        st.rerun()
