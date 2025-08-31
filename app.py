import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import torch


def get_text(pdf_doc):
    text = ''
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for pages in pdf_reader.pages:
            text+=pages.extract_text()
    return text        

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    text_chunks = text_splitter.split_text(text)
    return text_chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local('faiss_index')

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"])
    
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",            # will use GPU if available
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=512,
        temperature=0.3,
        top_p=0.9,
        return_full_text=False)
    
    llm = HuggingFacePipeline(pipeline=pipe)
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    #doc_chain = create_stuff_documents_chain(llm, prompt)

    return chain

def user_input(user_question):
    # Use HuggingFace embeddings (must be the same model used for index creation)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load FAISS index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Retrieve relevant documents
    docs = new_db.similarity_search(user_question, k=3)

    # Build the conversational QA chain with Qwen
    chain = get_conversational_chain()

    # Run the chain
    response = chain({"input_documents": docs, "question": user_question},return_only_outputs=True)
    #response = chain.invoke({"input_documents": docs, "question": user_question})

    # Print + streamlit output
    print("Reply:", response["output_text"])
    st.write("Reply:", response["output_text"])

def main():
    st.set_page_config("Chat with PDF")
    st.header("Chat with PDF using open source model")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()    
