import streamlit as st
import os
import time
from utils import (
    load_and_chunk_pdf,
    create_vector_store,
    save_user_data,
    load_user_data,
    authenticate_user,
)
from qa_chain import create_langgraph_chain

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 Chat with Your Research Paper")

SESSION_TIMEOUT = 60 * 10  # 10 minutes

# --- Login system with timeout ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.login_time = None

# Logout button
if st.session_state.authenticated:
    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

# Session timeout
if st.session_state.authenticated and st.session_state.login_time:
    if time.time() - st.session_state.login_time > SESSION_TIMEOUT:
        st.warning("Session expired. Please log in again.")
        st.session_state.clear()
        st.rerun()

if not st.session_state.authenticated:
    st.subheader("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state.authenticated = True
            st.session_state.user_id = username
            st.session_state.login_time = time.time()
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

# Load user's saved state if available
user_id = st.session_state.user_id
user_dir = os.path.join("user_data", user_id)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_user_data(user_dir, "chat_history") or []
    st.session_state.vectorstore = load_user_data(user_dir, "vectorstore")
    st.session_state.chain = (
        create_langgraph_chain(st.session_state.vectorstore)
        if st.session_state.vectorstore
        else None
    )

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file and uploaded_file.name != st.session_state.get("last_uploaded"):
    st.session_state["last_uploaded"] = uploaded_file.name
    os.makedirs(user_dir, exist_ok=True)
    pdf_path = os.path.join(user_dir, "document.pdf")
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing PDF..."):
        chunks = load_and_chunk_pdf(pdf_path)
        vectorstore = create_vector_store(chunks)
        chain = create_langgraph_chain(vectorstore)
        st.session_state.vectorstore = vectorstore
        st.session_state.chain = chain
        st.session_state.chat_history = []

        # Save for reuse
        save_user_data(user_dir, "vectorstore", vectorstore)
        save_user_data(user_dir, "chat_history", [])

    st.success("PDF processed. Start chatting below.")

if st.session_state.chain:
    chat_container = st.container()
    for i, (q, a) in enumerate(st.session_state.chat_history):
        with chat_container:
            with st.chat_message("user", avatar="👤"):
                st.markdown(q)
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(a)

    if prompt := st.chat_input("Ask a question about the document"):
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            result = st.session_state.chain.invoke(
                {"question": prompt, "chat_history": st.session_state.chat_history}
            )

        response = result["answer"]
        st.session_state.chat_history = result["chat_history"]
        save_user_data(user_dir, "chat_history", st.session_state.chat_history)

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(response)
