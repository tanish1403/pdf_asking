import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
google_api_key = os.getenv("AIzaSyCU4naBgeVjfrE6l9Vrz7vZCeTDa-VCSkA")
os.environ["google_api_key"] = "AIzaSyCU4naBgeVjfrE6l9Vrz7vZCeTDa-VCSkA"
genai.configure(api_key=google_api_key)


def convert_to_txt(pdf_path):
    text = ""
    for pdf in pdf_path:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def convert_text_chunk(text):
    splitter = RecursiveCharacterTextSplitter()
    return splitter.split_text(text)

def convert_to_vector(text):
    embed_model = GoogleGenerativeAIEmbeddings( model="models/embedding-001")
    vectors = FAISS.from_texts(text, embed_model)
    vectors.save_local("faiss_index")


# embed_model = GoogleGenerativeAIEmbeddings( model="models/embedding-001")
# new_db = FAISS.load_local("faiss_index", embed_model)
# print(new_db.search("power", 5))

def ask_question():
    test_prompt = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client = genai,
                                   temperature = 0.3)
    prompt = ChatPromptTemplate(template=test_prompt,input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

    


# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

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
    
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = ask_question()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    print(response)
    return response

with st.sidebar:
        st.title("Upload:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, type="pdf")
        if st.button("Start"):
            with st.spinner("Processing..."):
                raw_text = convert_to_txt(pdf_docs)
                text_chunks = convert_text_chunk(raw_text)
                convert_to_vector(text_chunks)
                st.success("Start Asking Questions")
st.title("Open pdfs and ask questions :book:")
if pdf_docs:
    #  st.write("You have uploaded", len(pdf_docs), "pdfs")
     st.write(f"You are asking from pdf : {pdf_docs[0].name}")
# st.write("Welcome to the chat!")
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "upload any book or pdf and ask"}]
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
                
                response = user_input(prompt)

                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)

