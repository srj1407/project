"""
A Strands agent that gets its tools from an MCP server, powered by Groq.

Same as agent_with_local_mcp.py; only the model provider changes. Groq exposes
an OpenAI-compatible API, so Strands' OpenAIModel talks to it via `base_url`.

Run:  python agent_with_local_mcp_groq.py      (needs GROQ_API_KEY, or a .env file)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp import StdioServerParameters, stdio_client
from strands import Agent
from strands.models.gemini import GeminiModel
from strands.tools.mcp import MCPClient

load_dotenv()

MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

model = GeminiModel(
    client_args={
        "vertexai": True,
    },
    model_id=MODEL_ID,
)

SERVER = Path(__file__).resolve().parent / "server.py"

# MCPClient takes a function that opens the transport. Here: spawn the server.
mcp_client = MCPClient(
    lambda: stdio_client(StdioServerParameters(command=sys.executable, args=[str(SERVER)]))
)

with mcp_client:
    tools = mcp_client.list_tools_sync()
    print("Tools discovered over MCP:", [t.tool_name for t in tools])

    agent = Agent(model=model, tools=tools)
    agent("I'm flying from Bangalore to London tomorrow. What's the weather like in both cities?")
    print()
