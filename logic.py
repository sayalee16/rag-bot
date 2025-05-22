from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

file_dir = "data/"
load_dotenv()

def get_answer(query, file):
    upload_pdf(file)
    pdf_path = file_dir + file.name
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20, add_start_index=True)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": len(chunks)})
    context = retriever.invoke(query)
    
    llm_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    response = llm_model.invoke(prompt)
    
    return response.content

def upload_pdf(file):
    with open(file_dir + file.name, "wb") as f:
        f.write(file.getbuffer())