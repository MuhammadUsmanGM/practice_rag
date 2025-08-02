"""
Example: Agent with Custom Session Backend (Redis, PostgreSQL, or Supabase)

Demonstrates how to use a production-ready session memory with the OpenAI Agents SDK.
Choose your backend by uncommenting the relevant import and initialization.

See: https://openai.github.io/openai-agents-python/sessions/
"""

import asyncio
import os

import google.generativeai as genai
from dotenv import load_dotenv
from agents import Agent, Runner
from agents import (
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_default_openai_client,
    set_tracing_disabled,
    set_default_openai_api,
    function_tool,
)

from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams

from dotenv import load_dotenv
load_dotenv()

#---------------------------------------------------------------

gemini_api_key = os.getenv("GEMINI_API_KEY")
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
set_default_openai_client(external_client)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash", openai_client=external_client
)
set_tracing_disabled(True)
set_default_openai_api("chat_completions")

# --- Qdrant setup ---
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
COLLECTION_NAME = "gemini_vectors"  # must match the one already created


# --- Embedding function using Gemini ---
def embed_query(text: str) -> list[float]:
    if not text.strip():
        raise ValueError("Query text cannot be empty")

    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_query",
    )
    return response["embedding"]


# --- Real Qdrant search tool ---
@function_tool(is_enabled=True)
def search_vector_db(query: str) -> str:
    """Search Panaversity vector DB and return top documents."""
    try:
        vector = embed_query(query)

        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=3,
            search_params=SearchParams(hnsw_ef=128),
        )

        if not results:
            return f"No relevant information found for: {query}"

        output = []
        for i, result in enumerate(results, 1):
            content = result.payload.get("content", "No content found")
            score = result.score
            source = result.payload.get("source_url", "Unknown")
            output.append(
                f"Result {i} (Score: {score:.3f}):\nSource: {source}\n{content}\n"
            )

        return "\n".join(output)

    except Exception as e:
        return f"Error during vector DB search: {e}"



#-------------------------------------------------------------


agent = Agent(
    name="PanaversityAssistant",
    instructions="""
You are a helpful assistant for Panaversity. Use the `search_vector_db` tool for questions about programs, courses, admissions, curriculum, or policies.
Do not guess — always rely on database info. Be friendly, clear, and accurate.
if you didn't find answer in database just say sorry to the user do not hallucinate.
""",
    tools=[search_vector_db],
    model=model,
)



#-----------------------------------------------------------------------




async def main():

    print("User: which courses are offered by panaversity?")
    result = await Runner.run(
        agent, "which courses are offered by panaversity?",
    )
    print(f"Assistant: {result.final_output}\n")


    print("=== Conversation Complete ===")


if __name__ == "__main__":
    asyncio.run(main())