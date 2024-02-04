import streamlit as st
from chat_pdf import convert_text_chunk, convert_to_vector, ask_question, convert_to_txt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
google_api_key = os.getenv("AIzaSyCU4naBgeVjfrE6l9Vrz7vZCeTDa-VCSkA")
os.environ["google_api_key"] = "AIzaSyCU4naBgeVjfrE6l9Vrz7vZCeTDa-VCSkA"
genai.configure(api_key=google_api_key)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

st.set_page_config(
        page_title="Ask your pdfs",
        page_icon=":book:"
    )
# with st.sidebar:
#     st.title("Ask your pdfs")
#     pdf = st.file_uploader("Upload your pdf", type="pdf", accept_multiple_files=True)
#     if st.button("Submit"):
#         text = convert_to_txt(pdf)
#         chunks = convert_text_chunk(text)

def clear_chat_history():
    """
    Clear the chat history by resetting the messages to an empty list.
    """
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]
    


with st.sidebar:
        st.title("Upload:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Start"):
            # with st.spinner("Processing..."):
            raw_text = convert_to_txt(pdf_docs)
            text_chunks = convert_text_chunk(raw_text)
            convert_to_vector(text_chunks)
            st.success("Start Asking Questions")
st.title("Open pdfs and ask questions :book:")
# st.write("Welcome to the chat!")
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload any book or pdf and ask"}]
for message in st.session_state.messages:
     with st.chat_message(message["role"]):
        st.write(message["content"])
if prompt :=st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)


if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                db =  FAISS.load_local("faiss_index", embeddings)
                docs = db.similarity_search(prompt)
                chain = ask_question()
                response = chain({"input_documents": docs, "question": prompt}, return_only_outputs=True, )

                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)

