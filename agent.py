### COMPLETE THE CODE ###
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from rag_tool import search_university_knowledge

model = ChatOllama(
    model="qwen3:0.6b",
    temperature=0,
    base_url="http://127.0.0.1:11434"
)

agent = create_agent(
    model=model,

    tools=[
        search_university_knowledge
    ],

    system_prompt="""
    You are a university IT support assistant.

    You MUST always call the search_university_knowledge
    tool before answering any question. Do not answer
    from memory. Base your answer only on the content
    returned by the tool.

    The policy documentation does NOT contain
    phone numbers, email addresses, or any other
    contact details for the IT Service Desk.

    If the user asks for the IT Service Desk phone
    number, or any contact detail that is not in
    the policy documentation, you MUST say that
    the exact information is not available from
    the university policy documentation.

    Never invent phone numbers, email addresses,
    URLs, names, or any other contact details.

    Do not guess.
    """
)


question = """
What exact phone number should I call for
the University IT Service Desk?
"""
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": question
        }
    ]
})


print("\nFINAL ANSWER")
print(result["messages"][-1].content)

print("\n\nFULL AGENT TRACE")
for message in result["messages"]:
    print("\n--------------------")
    print(type(message).__name__)
    print(message)