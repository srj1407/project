import os
import sys

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
from langchain_tavily import TavilySearch  # noqa: E402

from agent_trace_printer import print_agent_trace  # noqa: E402

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

DEFAULT_QUESTION = (
    "Find 3 upcoming AI or developer meetups in Bengaluru and give the date, "
    "venue and a link for each."
)

llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, vertexai=True)

# A ready-made LangChain tool. Like TavilyClient, it reads TAVILY_API_KEY from
# the environment. You can set defaults here, e.g. TavilySearch(max_results=5).
search_tool = TavilySearch()
tools = [search_tool]
agent = create_agent(model=llm, tools=tools)


def main() -> None:
    print("Hello from Future with Shashwat — AI Harness!")
    print(f"Tool name the LLM sees: {search_tool.name}")

    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    print(f"Question: {question}")

    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    print_agent_trace(result)


# ↑ Peek at the vendor's "menu card" yourself:
#   uv run python -c "from langchain_tavily import TavilySearch; import os; os.environ.setdefault('TAVILY_API_KEY','x'); t=TavilySearch(); print(t.name); print(t.args)"
#   → tavily_search
#   → {'query': {...}, 'include_domains': {...}, 'exclude_domains': {...}, 'search_depth': {...},
#      'include_images': {...}, 'time_range': {...}, 'topic': {...}, 'start_date': {...}, 'end_date': {...}}


if __name__ == "__main__":
    main()
