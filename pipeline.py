from tqdm import tqdm
import chromadb
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent

# ---------- INITIALIZE COMPONENTS ----------
def initialize_components():
    embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")
    model = init_chat_model("google_genai:gemini-2.5-flash-lite")
    cloud_client = chromadb.CloudClient(
        api_key="ck-AG66yDL8KK4uwvsq8wtXNjz1MUr4VueutwZisyAq4ThT",
        tenant="9682eaae-d806-426c-b9bd-2f0682eec51a",
    )
    vector_store = Chroma(
        client=cloud_client,
        collection_name="new_worldv1",
        embedding_function=embeddings,
    )
    return embeddings, model, cloud_client, vector_store


# ---------- DATA PREPARATION ----------
def load_documents(file_paths):
    all_docs = []
    for path in file_paths:
        loader = CSVLoader(file_path=path)
        docs = loader.load()
        all_docs.extend(docs)
    print(f"✅ Loaded {len(all_docs)} documents.")
    return all_docs


def split_documents(all_docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
    splits = splitter.split_documents(all_docs)
    print(f"✅ Split into {len(splits)} chunks.")
    return splits


def add_documents_to_chroma(vector_store, splits, batch_size=300):
    ids = []
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i + batch_size]
        batch_ids = vector_store.add_documents(documents=batch)
        ids.extend(batch_ids)
        print(f"✅ Added batch {i // batch_size + 1}")
    return ids


# ---------- AGENT CREATION ----------
def create_query_agent(model, vector_store):
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information to help answer a query."""
        docs = vector_store.similarity_search(query, k=1)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in docs
        )
        return serialized, docs

    tools = [retrieve_context]
    prompt = "You are an assistant that answers questions about household data."
    agent = create_agent(model, tools, system_prompt=prompt)
    return agent
