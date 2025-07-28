from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
# Load data
loader = CSVLoader(file_path="assets/health_forum.csv", encoding="utf-8", csv_args={"delimiter": ","})
documents = loader.load()

# Split document to chuck
splitter = CharacterTextSplitter(separator="\n",
                                 chunk_size=500,  # Thai sentences are often longer
                                 chunk_overlap=50)
split_docs = splitter.split_documents(documents)

# convert text to vector with HuggingFace
embedding = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True})

# store vector in FAISS
vectorstore = FAISS.from_documents(split_docs, embedding)

# save vector
vectorstore.save_local("assets/vector_store_ch500")
