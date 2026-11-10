import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load the API key from the .env file
load_dotenv()

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🏢")
st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("Ask me anything about HR policies, leaves, and workplace guidelines.")

# 1. Build and cache the FAISS database locally in batches
@st.cache_resource(show_spinner="Building HR database (this takes a minute to respect API limits)...")
def initialize_rag_pipeline():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # If the database is already saved locally, load it instantly
    if os.path.exists("faiss_index"):
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        # Load PDFs from the 'data' folder
        loader = PyPDFDirectoryLoader("data")
        docs = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)
        
        # Start the database with the first 5 chunks
        vector_store = FAISS.from_documents(splits[:5], embeddings)
        
        # Loop through the rest of the chunks slowly to avoid the 429 quota error
        for i in range(5, len(splits), 5):
            time.sleep(5)  # Pause for 5 seconds between batches
            vector_store.add_documents(splits[i:i+5])
            
        # Save it permanently so we never have to wait for this again
        vector_store.save_local("faiss_index")
    
    # Setup the MMR retriever
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.75}
    )
    
    return retriever

retriever = initialize_rag_pipeline()

# 2. Setup the Gemini LLM and Guardrail Prompt
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.1)

guardrail_template = """You are the official HR Help Desk Assistant for Zyro Dynamics Pvt. Ltd.

Follow these strict rules when answering:
1. OUT-OF-SCOPE GUARDRAIL: If the user's question is unrelated to HR policies, workplace rules, employee benefits, or Zyro Dynamics, do NOT answer it. Instead, reply ONLY with:
"I am sorry, but I can only answer questions related to Zyro Dynamics HR policies and employee guidelines."

2. MISSING POLICY INFORMATION: If the question is related to HR or workplace policies, but the answer is not present in the provided Context below, do NOT guess or invent rules. Reply with:
"I could not find information regarding this in the available Zyro Dynamics HR policy documents."

3. GROUNDED IN-SCOPE ANSWER: Answer strictly using ONLY the facts, numbers, dates, grades, and conditions from the Context below. Keep the answer clear, concise, and in plain text.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(guardrail_template)

# Helper function to format documents and extract citations
def format_docs_with_citations(docs):
    formatted_text = []
    citations = set()
    for doc in docs:
        formatted_text.append(doc.page_content)
        filename = os.path.basename(doc.metadata.get('source', 'Unknown Document'))
        page = doc.metadata.get('page', 0) + 1 
        citations.add(f"{filename} (Page {page})")
    
    context_string = "\n\n".join(formatted_text)
    return context_string, list(citations)

# 3. Setup Streamlit Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Handle New User Input
if user_question := st.chat_input("E.g., How many days of casual leave do I get?"):
    st.chat_message("user").markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            try:
                retrieved_docs = retriever.invoke(user_question)
                context_string, citations_list = format_docs_with_citations(retrieved_docs)
                
                chain = prompt | llm | StrOutputParser()
                bot_answer = chain.invoke({"context": context_string, "question": user_question})
                
                final_response = bot_answer
                if "I am sorry, but I can only answer questions related to" not in bot_answer and "I could not find information" not in bot_answer:
                    citation_text = "\n\n**Sources:**\n" + "\n".join([f"- {cite}" for cite in citations_list])
                    final_response += citation_text
                
                st.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                
            except Exception as e:
                error_msg = f"Error processing query: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})