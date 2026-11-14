import os

from dotenv import load_dotenv

# Load keys BEFORE importing/creating anything that needs them.
load_dotenv()

from langchain.agents import create_agent  # noqa: E402
from langchain.tools import tool  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_google_genai import ChatGoogleGenerativeAI

from agent_trace_printer import print_agent_trace  # noqa: E402

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

QUESTION = "What's the weather like in Patna right now?"


# -----------------------------------------------------------------------------
# search()  —  a LangChain TOOL
# -----------------------------------------------------------------------------
# @tool turns this ordinary function into something an agent can call.
# LangChain reads three things and sends them to the LLM as the tool's
# "menu card":
#   • the NAME        → "search"
#   • the TYPE HINTS  → query: str, returns str
#   • the DOCSTRING   → when to use the tool and what each argument means
# The LLM decides WHETHER and HOW to call the tool purely from that card, so
# the docstring is written for the MODEL: say clearly when the tool helps.
#
#   Input : query (str) — written by the LLM, e.g. "Bengaluru weather now"
#   Output: (str) the "search result" (hard-coded for now)
@tool
def search(query: str) -> str:
    """Look up live information on the internet.

    Use this for anything that changes over time and that you cannot know
    from memory: today's weather, recent news, current prices, upcoming events.

    Args:
        query: What to look up, as a short search phrase.

    Returns:
        The most relevant result as plain text.
    """
    print(f"🔎 search tool called with: {query}")
    return "Patna right now: 24°C, light drizzle, humidity 78%"


# ↑ You can call a tool directly too (handy for testing):
#   search.invoke({"query": "anything at all"})
#   → prints   🔎 search tool called with: anything at all
#   → returns  "Patna right now: 24°C, light drizzle, humidity 78%"
#   search.name        → "search"
#   search.description → "Look up live information on the internet. ..."


# -----------------------------------------------------------------------------
# Build the agent: a reasoning engine (LLM) + a list of tools
# -----------------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        vertexai=True
    )
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main() -> None:
    print("Hello from Future with Shashwat — AI Harness!")

    # An agent's input is a dict with a "messages" list — the same shape as a
    # chat conversation. HumanMessage = "this was typed by the user".
    result = agent.invoke({"messages": [HumanMessage(content=QUESTION)]})

    # `result` is a dict: {"messages": [HumanMessage, AIMessage, ToolMessage, AIMessage]}
    # print(result)          # ← uncomment to see the raw dict
    print_agent_trace(result)


if __name__ == "__main__":
    main()
