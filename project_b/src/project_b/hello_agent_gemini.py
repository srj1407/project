import os

from dotenv import load_dotenv
from strands import Agent
from strands.models.gemini import GeminiModel

load_dotenv()

# One place to choose the model. Override without editing code:
#   export GEMINI_MODEL=gemini-3.8-flash
MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

model = GeminiModel(
    client_args={
        "vertexai": True,
    },
    model_id=MODEL_ID,
)
agent = Agent(model=model)

response = agent("Hello! Tell me a fun fact about AI agents.")
print()
print("-" * 60)
print(response)
