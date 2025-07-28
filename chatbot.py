from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
def load_chain():
    embedding = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True})
    vectorstore = FAISS.load_local("assets/vector_store_ch500", embedding, allow_dangerous_deserialization=True)
    # convert vectorstore to retriever object for searching information that closely question
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",  # or "similarity_score_threshold", "mmr"
                                         search_kwargs={"score_threshold": 0.7} ) 
                                        # search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.7})# for mmr
    
    # - You must greet politely with the patients only the starting conversation. - Your answer only the current question.
    # create prompt
    prompt = PromptTemplate(
    input_variables=["context", "question"],
    template = """
    You are a helpful, professional, and empathetic AI medical assistant. 
    Importantly, use the provided context from patient-related data to answer the user's question as clearly and accurately as possible.
    Additionally, please follow the strict instructions:
    <instructions>
    - You must respond in the same language every response as the patients' language with clear, and formal languages.
    - You must ONLY use information from the context. DO NOT guess or fabricate any information.
    - Your answer must not refer to the source of informatiom.
    - Your answer must be concise and focused only on the current question. DO NOT duplicate the previous answer.
    - If the context does not contain the answer, say: "I'm sorry, but I cannot provide an answer based on the available information."
    - You must close conversation politely when they do not have any questions.
    </instructions>

    Conversation history:
    {chat_history}

    Relevant patient information or documentation:
    {context}

    Question from patient:
    {question}

    Your response:
    """)

    llm = ChatOpenAI(openai_api_base="https://api.groq.com/openai/v1",
                    openai_api_key=OPENAI_API_KEY,
                    max_tokens=500, # increase
                    model_name="llama3-70b-8192", #llama3-8b-8192, llama3-70b-8192
                    temperature=0.3)
    # create memory to store history chat
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                    retriever=retriever,
                                                    memory=memory,
                                                    combine_docs_chain_kwargs={"prompt": prompt})
    
    return qa_chain

# create chatbot with streamlit
st.set_page_config(page_title="Agnos Chatbot", page_icon="🩺")
st.title("🩺 Agnos Chatbot")

if "chain" not in st.session_state:
    st.session_state.chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input box
if prompt := st.chat_input("Ask me anything about the docs..."):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response from bot
    response = st.session_state.chain.run(prompt)
    st.chat_message("assistant").write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})