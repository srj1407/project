"""
A Strands agent connected to the AWS Documentation MCP server, powered by Groq.

Same as agent_with_aws_docs_mcp.py; only the model provider changes. No AWS
credentials are needed: the docs server reads public documentation.

Prerequisite: `uv` installed (it provides the `uvx` command).

Run:  python agent_with_aws_docs_mcp_groq.py    (needs GROQ_API_KEY, or a .env file)
"""

import os
import shutil

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

if shutil.which("uvx") is None:
    raise SystemExit("`uvx` not found. Install uv first: pip install uv")


# Create MCP client
mcp_client = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="uvx",
            args=["awslabs.aws-documentation-mcp-server@latest"],
            env={**os.environ, "FASTMCP_LOG_LEVEL": "ERROR"},
        )
    ),
    startup_timeout=120,  # first run downloads the server package
)

# Use with agent
with mcp_client:
    tools = mcp_client.list_tools_sync()
    print("Tools from the AWS Documentation MCP server:", [t.tool_name for t in tools])

    agent = Agent(
        model=model,
        tools=tools,
        system_prompt="Answer AWS questions using the documentation tools. Cite the doc URLs you used.",
    )
    agent("What is AWS Lambda, and what is the maximum timeout for a Lambda function?")
    print()
