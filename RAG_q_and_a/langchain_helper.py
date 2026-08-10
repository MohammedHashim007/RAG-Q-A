import os

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


load_dotenv()

# Get Google API key
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not configured.")


# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=google_api_key,
    temperature=0.1
)


# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectordb_file_path = "faiss_index"


def create_vector_db():

    loader = CSVLoader(
        file_path="Q&A.csv",
        source_column="prompt"
    )

    data = loader.load()

    vectordb = FAISS.from_documents(
        documents=data,
        embedding=embeddings
    )

    vectordb.save_local(vectordb_file_path)


def get_qa_chain():

    vectordb = FAISS.load_local(
        vectordb_file_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.7,
            "k": 3
        }
    )

    prompt_template = """
You are a helpful FAQ assistant.

Answer the question using ONLY the information
provided in the context.

If the answer cannot be found in the context,
say "I don't know."

Do not make up information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PROMPT
        }
    )

    return chain


if __name__ == "__main__":

    create_vector_db()

    chain = get_qa_chain()

    response = chain.invoke({
        "query": "Do you have javascript course?"
    })

    print(response["result"])
