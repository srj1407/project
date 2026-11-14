import os
import sys

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent  # noqa: E402
from langchain.tools import tool  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
from tavily import TavilyClient  # noqa: E402

from agent_trace_printer import print_agent_trace  # noqa: E402

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

DEFAULT_QUESTION = (
    "Find 3 upcoming AI or developer meetups in Bengaluru and give the date, "
    "venue and a link for each."
)

# The raw Tavily SDK client. It reads TAVILY_API_KEY from the environment.
tavily = TavilyClient()


# -----------------------------------------------------------------------------
# search()  —  our own tool, now backed by real web search
# -----------------------------------------------------------------------------
# Same name, same docstring, same argument as in Script 01, so the LLM sees the
# same "menu card". Only the implementation changed.
#
#   Input : query (str) e.g. "upcoming AI meetups Bengaluru"
#   Output: (dict) Tavily's JSON answer, roughly:
#           {"query": "...",
#            "results": [{"title": "...", "url": "https://...",
#                         "content": "<snippet>", "score": 0.93}, ...],
#            "response_time": 1.2, ...}
#   (The return type is a dict, not a str — LangChain turns it into JSON text
#    for the LLM automatically.)
@tool
def search(query: str) -> dict:
    """Look up live information on the internet.

    Do not submit queries containing only site: operators. Always include
    descriptive search terms, such as "event details site:luma.com".

    Use this for anything that changes over time and that you cannot know
    from memory: today's weather, recent news, current prices, upcoming events.

    Args:
        query: What to look up, as a short search phrase.

    Returns:
        The most relevant results, each with a title, URL and short snippet.
    """
    print(f"🔎 search tool called with: {query}")
    result = tavily.search(query=query)
    print(f"🔎 search tool returned {result}")
    return result  # tip: tavily.search(query=query, max_results=5)


# ↑ Example:
#   search.invoke({"query": "Bengaluru weather now"})
#   → {"query": "Bengaluru weather now",
#      "results": [{"title": "Weather in Bengaluru", "url": "https://www.weatherapi.com/", ...}, ...]}


llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, vertexai=True)
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main() -> None:
    print("Hello from Future with Shivank — AI Harness!")

    # sys.argv[1:] are the words typed after the script name, if any.
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    print(f"Question: {question}")

    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    print_agent_trace(result)


if __name__ == "__main__":
    main()
