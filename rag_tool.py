from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from policy_loader import load_policy_documents

documents = load_policy_documents()

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434",
)

vector_store = Chroma(
    collection_name="university_policies_agent",
    embedding_function=embeddings,
)

batch_size = 100
for i in range(0, len(documents), batch_size):
    chunk = documents[i : i + batch_size]
    vector_store.add_documents(chunk)

@tool
def search_university_knowledge(question: str) -> str:
    """
    Search the official university IT policy
    documentation for information about university
    procedures, instructions, and policies
    (printing, storage, accounts, vpn, wifi, email).

    Use this tool for any question about what the
    university's IT rules or procedures say.

    IMPORTANT: This tool does NOT contain phone
    numbers, email addresses, or any contact
    details for the IT Service Desk, because those
    are not part of the policy documentation.
    """

    results = vector_store.similarity_search(question, k=3)

    retrieved = []
    for document in results:
        retrieved.append(
            f"POLICY ID: {document.metadata['policy_id']}\n"
            f"TOPIC: {document.metadata['topic']}\n"
            f"{document.page_content}"
        )

    return "\n\n".join(retrieved)