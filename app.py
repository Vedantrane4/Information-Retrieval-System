import streamlit as st
from src.helper import (
    get_pdf_text,
    get_text_chunks,
    get_vector_store,
    get_conversational_chain
)


def user_input(user_question):
    if st.session_state.conversation is None:
        st.warning("Upload PDF first.")
        return

    try:
        response = st.session_state.conversation.invoke({
            "question": user_question
        })
    except Exception as e:
        st.error(f"Error: {e}")
        return

    chat_history = response["chat_history"]

    for i, msg in enumerate(chat_history):
        if i % 2 == 0:
            st.write("🧑 User:", msg.content)
        else:
            st.write("🤖 Bot:", msg.content)


def main():
    st.set_page_config(page_title="PDF Chatbot")
    st.header("📄 Chat with your PDF")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    user_question = st.text_input("Ask something...")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Upload PDFs")
        pdf_docs = st.file_uploader(
            "Upload PDFs",
            accept_multiple_files=True
        )

        if st.button("Process"):
            if not pdf_docs:
                st.warning("Upload at least one PDF")
                return

            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(chunks)

                    st.session_state.conversation = get_conversational_chain(vector_store)

                    st.success("Done ✅")

                except Exception as e:
                    st.error(f"Error: {e}")


if __name__ == "__main__":
    main()