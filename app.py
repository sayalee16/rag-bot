import streamlit as st
from logic import get_answer

st.header("RAG Based PDF Question Answering System")
file = st.file_uploader("Upload a file", type=["pdf"])
query = st.text_area("Enter your question here", placeholder="Ask me anything about the PDF file you uploaded.")
ask_question = st.button("Ask Question")

if ask_question:
    if file:
        st.chat_message("user", avatar="👤").write(query)
        answer = get_answer(query, file)
        st.chat_message("AI Assistant", avatar="🤖").write(answer)
    else:
        st.error("Please upload a PDF file to ask questions.")