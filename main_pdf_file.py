import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from textwrap import dedent


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def format_docs(docs) -> str:
    """Formatter function that joins documents into single string delimited by 2 newlines.

    Args:
        docs (List): List of Documents to be processed

    Returns:
        str: Newline joined documents
    """
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":

    load_dotenv()
    print("FAISS Vectorstore")
    pdf_path = os.environ.get("PDF_FILEPATH")
    # By default it will chunk it by page. May still be too large in context window
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    # Split further by character
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    ## Old code involving OpenAI service
    #embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    

    # Use of huggingface's Google t5-base as example
    embedding_model_id =  os.environ.get("HUGGINGFACEHUB_VECTORDB_EMBEDDING_MODEL_NAME")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_id, # Provide the pre-trained model's path
        model_kwargs={"device":DEVICE}, # Pass the model configuration options
        encode_kwargs = {'normalize_embeddings': False}
    )
    
    # Use FAISS Vector store to loaded chunkified docs into RAM with defined embeddings from Huggingface
    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=embeddings
    )
    vectorstore.save_local("faiss_index_react")

    # Load data from FAISS vector store
    new_vectorstore = FAISS.load_local(
        "faiss_index_react",
        embeddings=embeddings,
        allow_dangerous_deserialization=True #Allow deserialisation for trusted file.Otherwise it is not advised to. This is a feature to prevent any dangerous executions by default from a .pkl file
    )
    
    retriever = new_vectorstore.as_retriever(search_kwargs={"k": 4})
    # Use open sourced chat prompt as part of ReAct for QA use case. https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    query = "Give me the gist of ReAct in 3 sentences"

    # Query from vectorstore
    relevant_documents = new_vectorstore.similarity_search(query)
    print(f'There are {len(relevant_documents)} documents retrieved which are relevant to the query. Display the first one:\n')
    print(relevant_documents[0].page_content)

    print("Querying with ChatOpenAI default model with more deterministic output")
    llm_chatopenai = ChatOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-3.5-turbo",
        temperature=0
    )
    print()
    print("LCEL implementation with retrieval > chat prompt > LLM")
    print("----------------------------")
    # For retrieval dict, you need to identify the variables in the prompt. Contains variables context/input
    retrieval = {
        "context": vectorstore.as_retriever() | format_docs,
        "input": RunnablePassthrough(),
    }

    # Feed dictionary to retrieval qa_chat_prompt followed by LLM to parse.
    rag_pdf_chain_openai = (
        retrieval | retrieval_qa_chat_prompt | llm_chatopenai
    )

    # Invoke with require template input(chat prompt) and get result
    result = rag_pdf_chain_openai.invoke(input=query)
    print(result.content)
    print()
    # Similar to combining create_stuff_documents_chain and create_retrieval_chain
    print("RetrievalQA from LLM call without use of retrieval-qa-chat prompt provided. Only refine chain type included")
    print("-----------")

    qa = RetrievalQA.from_chain_type(
        llm=llm_chatopenai,
        chain_type="refine",
        retriever=retriever,
    )
    result = qa.invoke({"query": query})
    print(result["result"])
    print()
    print("RetrievalQA from LLM call with use of custom prompt")
    print("-----------")


    # Create Prompt
    template = dedent("""\
        Answer any use questions based solely on the context below:
        
        <context>
        
        {context}

        </context>
        
        Question: {question}
        Answer:"""
    )
    prompt = PromptTemplate.from_template(template)

    qa = RetrievalQA.from_chain_type(
        llm=llm_chatopenai,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},  # The prompt is added here
    )
    result = qa.invoke({"query": query})
    print(result["result"])
    print()

    # Huggingface experimentation
    llm_model_id = os.environ.get("HUGGINGFACEHUB_LLM_QA_MODEL_NAME")

    print("Case of HuggingFaceEndpoint (online)")
    print("-----------")
    callbacks = [StreamingStdOutCallbackHandler()]
    # initialize Hub LLM with model. Note the model size
    hub_llm = HuggingFaceEndpoint(
        repo_id = os.environ.get("HUGGINGFACEHUB_LLM_QA_MODEL_NAME"),
        temperature = 0.01,
        top_k = 5,
        huggingfacehub_api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        max_new_tokens = int(os.environ.get("HUGGINGFACEHUB_LLM_QA_MODEL_MAX_TOKEN")),
        callbacks = callbacks
    )

    qa = RetrievalQA.from_chain_type(
        llm=hub_llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},  # The prompt is added here
    )
    result = qa.invoke({"query": query})
    print(result["result"])
    print()
