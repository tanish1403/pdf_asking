import streamlit as st
from chat_pdf import convert_text_chunk, convert_to_vector, ask_question, convert_to_txt
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


with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = convert_to_txt(pdf_docs)
                text_chunks = convert_text_chunk(raw_text)
                convert_to_vector(text_chunks)
                st.success("Done")