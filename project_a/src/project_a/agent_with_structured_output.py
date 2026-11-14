import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from langchain.agents import create_agent  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
from langchain_tavily import TavilySearch  # noqa: E402

from agent_trace_printer import print_agent_trace  # noqa: E402

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

QUESTION = (
    "Find 3 upcoming AI or developer meetups in Bengaluru and give the date, "
    "venue and a link for each."
)


# -----------------------------------------------------------------------------
# The OUTPUT SCHEMA (Pydantic models)
# -----------------------------------------------------------------------------
# Pydantic classes describe the shape of the data we want back. The class
# docstrings and Field descriptions are sent to the LLM, so — just like tool
# docstrings — they are instructions for the model.


class Source(BaseModel):
    """A web page the answer is based on."""

    title: str = Field(description="Short title of the page, e.g. the event name")
    url: str = Field(description="Full link to the page, starting with https://")


# ↑ Example:  Source(title="Bengaluru AI Builders — October meetup",
#                    url="https://www.meetup.com/...")


class AgentAnswer(BaseModel):
    """The agent's final reply: a readable answer plus the pages it used."""

    answer: str = Field(description="The answer to the user's question, in plain text")
    sources: list[Source] = Field(
        default_factory=list,  # an empty list if the agent cites nothing
        description="Every web page used to write the answer",
    )


# ↑ Example:
#   AgentAnswer(answer="Here are 3 meetups ...",
#               sources=[Source(title="...", url="https://..."), ...])
#   .model_dump() → {"answer": "Here are 3 meetups ...",
#                    "sources": [{"title": "...", "url": "https://..."}, ...]}


llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, vertexai=True)

search_tool = TavilySearch()
tools = [search_tool]

# response_format=AgentAnswer is the ONLY change compared with Script 03.
agent = create_agent(model=llm, tools=tools, response_format=AgentAnswer)


def main() -> None:
    print("Hello from Future with Shashwat — AI Harness!")
    result = agent.invoke({"messages": [HumanMessage(content=QUESTION)]})

    # The usual message trail is still there ...
    print_agent_trace(result)

    # ... plus a NEW key holding a validated Pydantic object.
    structured: AgentAnswer = result["structured_response"]
    print("\n" + "=" * 60)
    print("STRUCTURED RESPONSE")
    print("=" * 60)
    print(f"type: {type(structured).__name__}")
    print("answer:")
    for line in structured.answer.splitlines():
        print(f"  {line}")
    print(f"sources ({len(structured.sources)}):")
    for source in structured.sources:
        print(f"  - {source.title}")
        print(f"    {source.url}")


if __name__ == "__main__":
    main()
