from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from app.config import setup_environment
from app.pipeline import (
    initialize_components,
    create_query_agent,
    load_documents,
    split_documents,
    add_documents_to_chroma,
)
import os
import tempfile
from langchain_chroma import Chroma

app = FastAPI(title="Household Facility RAG API")

# Global variables
agent = None
vector_store = None
cloud_client = None
embeddings = None
model = None


class QueryInput(BaseModel):
    query: str


@app.on_event("startup")
def startup_event():
    global agent, vector_store, cloud_client, embeddings, model

    print("🚀 Starting Household Facility RAG API...")
    setup_environment()

    embeddings, model, cloud_client, vector_store = initialize_components()

    COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "muneeb_collection")

    # ✅ Wrap cloud collection for LangChain usage
    vector_store = Chroma(
        client=cloud_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    agent = create_query_agent(model, vector_store)

    print(f"✅ Connected to Chroma Cloud collection '{COLLECTION_NAME}'")
    print("✅ API initialized and ready to handle requests!")


@app.on_event("shutdown")
def shutdown_event():
    print("🛑 Shutting down the API...")


# ------------------------------------------------------
# 🆕 CSV Upload Endpoint
# ------------------------------------------------------
@app.post("/upload_csv")
async def upload_csv(
    file: UploadFile = File(...),
    collection_name: str = Form(...),
):
    """
    Upload a CSV file, load it, embed it, and create a new Chroma Cloud collection.
    """
    global embeddings, cloud_client

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        print(f"📂 Uploaded CSV saved temporarily at {tmp_path}")

        # Load and split documents
        all_docs = load_documents([tmp_path])
        splits = split_documents(all_docs)

        # ✅ Create new Chroma Cloud collection
        new_collection = Chroma(
            client=cloud_client,
            collection_name=collection_name,
            embedding_function=embeddings,
        )

        # Add embedded docs to Chroma
        add_documents_to_chroma(new_collection, splits)

        print(f"✅ New collection '{collection_name}' created and populated in Chroma Cloud!")

        
        #  Switch global vector_store and agent to use the new collection
        global vector_store, agent
        vector_store = new_collection
        agent = create_query_agent(model, vector_store)

        print(f"🔄 Switched active collection to '{collection_name}' for future queries")

        return {"message": f"✅ Collection '{collection_name}' successfully created and now active for querying."}

    except Exception as e:
        return {"error": f"⚠️ Failed to process file: {e}"}


# ------------------------------------------------------
# Existing endpoints
# ------------------------------------------------------
@app.post("/ask")
def ask_question(input_data: QueryInput):
    """Handle natural language queries."""
    global agent

    if agent is None:
        return {"error": "❌ Agent not initialized yet. Please restart the server."}

    query = input_data.query
    response_text = ""

    try:
        for event in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            response_text = event["messages"][-1].content

        return {"query": query, "response": response_text}

    except Exception as e:
        return {"error": f"⚠️ Query failed: {e}"}


@app.get("/")
def home():
    """Health check endpoint."""
    return {"message": "✅ RAG API is running! Use POST /ask to query or POST /upload_csv to upload new data."}
