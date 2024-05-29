import time
from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain

class TextSplitter:
  def __init__(self):
      self.docs = None

  def split_text(self, data: str) -> List[str]:
      if not data:
          return []
      recur_text_splitter = RecursiveCharacterTextSplitter(
          separators=['\n', '\n\n', '\n\n\n', '\n\n\n\n', '\n\n\n\n\n', '.', ',', ],
          chunk_size = 1000
      )
      docs = recur_text_splitter.split_documents(data)
      return docs

class MainApp:
  def __init__(self):
    load_dotenv()
    self.main_container = st.empty()
    self.llm = OpenAI(temperature=0.2, max_tokens=500)
    self.text_splitter = TextSplitter()

  def run(self):
    # Main title / description
    st.title("sourceIt")
    text_placeholder = st.empty()
    text_placeholder.text("Ask questions, get answers and their sources.")
    
    # Sidebar + button
    st.sidebar.title("Article Urls")
    urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
    input_url_button = st.sidebar.button("Submit")
    
    if input_url_button:  
      text_placeholder.text("Loading Url(s)...")
      time.sleep(0.5)
      text_placeholder.text("Splitting text...")
      time.sleep(0.5)
      text_placeholder.text("Creating vector embeddings...")
      time.sleep(0.5)
      text_placeholder.text("Storing embeddings to vector database...")
      time.sleep(0.5)
      text_placeholder.text("Embeddings stored in vector database.")
      time.sleep(0.5)
      text_placeholder.text("Loading...")

      time.sleep(2)
      text_placeholder.empty()
      st.write("SourceIt: Hello 👋")
      st.write("SourceIt: What would you like to know from the article(s) you provided?")
    
    # Load embeddings and split text
    embeddings = OpenAIEmbeddings()
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    split_texts = self.text_splitter.split_text(data)

    # Create FAISS vector index
    index = FAISS.from_documents(split_texts, embeddings)
    index.save_local("/Path/To/Your/Dir")

    # Display chat messages
    if "messages" not in st.session_state:
      st.session_state.messages = []

    for message in st.session_state.messages:
      with st.chat_message(message["role"]):
        st.markdown(message["content"])

    prompt = st.chat_input("Message sourceIt...")

    # Receive user input and generate response
    if prompt:
      with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

      # Generate response with OpenAI model and FAISS index
      with st.chat_message("sourceIt"):
        chain = RetrievalQAWithSourcesChain.from_llm(llm=self.llm, retriever=index.as_retriever())
        result = chain.invoke({"question": prompt}, return_only_outputs=True)
        
        answer = result["answer"]
        sources = result.get("sources", "")
        combined_message = f"{answer}\n\n"
        if sources:
            combined_message += "Source(s):\n"
            sources_list = sources.split("\n")
            for source in sources_list:
                combined_message += f"- {source}\n"

        st.markdown(combined_message)
        st.session_state.messages.append({"role": "sourceIt", "content": combined_message})

if __name__ == "__main__":
    app = MainApp()
    app.run()