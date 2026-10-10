from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from policy_loader import load_policy_documents

documents = load_policy_documents()

print(f"Loaded {len(documents)} policy documents.")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434",
)

vector_store = Chroma(
    collection_name="university_policies",
    embedding_function=embeddings,
)

batch_size = 100
for i in range(0, len(documents), batch_size):
    chunk = documents[i : i + batch_size]
    vector_store.add_documents(chunk)
    print(f"Added {i + len(chunk)} / {len(documents)}")

question = (
    "What exact phone number should I call "
    "for the University IT Service Desk?"
)

results = vector_store.similarity_search(question, k=3)

print("\nRETRIEVED DOCUMENTS")
for document in results:
    print("\n--------------------")
    print("Policy ID:", document.metadata["policy_id"])
    print("Topic:", document.metadata["topic"])
    print(document.page_content)

context = "\n\n".join(
    document.page_content for document in results
)

model = ChatOllama(
    model="qwen3:0.6b",
    temperature=0,
    base_url="http://127.0.0.1:11434",
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a university IT support assistant.

        Answer the user's question using ONLY the
        supplied university policy documentation.

        If the documentation does not contain the
        exact information requested, you MUST say
        that the information is not available.

        Never invent phone numbers, email addresses,
        URLs, names, or any other contact details.

        Do not guess.
        """
    ),
    (
        "human",
        """
        UNIVERSITY POLICY DOCUMENTATION:

        {context}

        QUESTION:

        {question}
        """
    ),
])

rag_chain = prompt | model

response = rag_chain.invoke({
    "context": context,
    "question": question,
})

print("\nFINAL ANSWER")
print(response.content)