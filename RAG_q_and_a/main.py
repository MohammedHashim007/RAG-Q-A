import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.title("Retrieval-Augmented Generation (RAG) Q&A 🌱")

btn = st.button("Create Knowledgebase")
if btn:
    with st.spinner("Creating vector database from FAQ sheet..."):
        create_vector_db()
    st.success("Knowledgebase created successfully!")

question = st.text_input("Question: ")

if question:
    with st.spinner("Searching knowledgebase and generating answer..."):
        chain = get_qa_chain()
        # Use invoke with the 'query' key matching input_key in langchain_helper
        response = chain.invoke({"query": question})

    st.header("Answer")
    st.write(response["result"])
