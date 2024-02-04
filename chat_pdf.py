from PyPDF2 import PdfReader
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


def convert_to_txt(pdf_path, start_page=0, end_page=None):
    reader = PdfReader(pdf_path)
    if end_page is None:
        end_page = len(reader.pages)
    text = ""
    for i in range(start_page, end_page):
        text += reader.pages[i].extract_text()
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

def ask_question(question):
    test_prompt = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-ultra",
                                   client = genai.client(),
                                   temperature = 0.5)
    prompt = PromptTemplate(test_prompt,input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

    
