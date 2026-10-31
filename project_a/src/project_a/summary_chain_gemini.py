import os

from dotenv import load_dotenv

# Load OPENAI_API_KEY (and optional LANGSMITH_* settings) from .env.
load_dotenv()

from langchain_core.prompts import PromptTemplate  # noqa: E402
from langchain_google_genai import ChatGoogleGenerativeAI

from sample_data import PERSON_INFORMATION  # noqa: E402

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# -----------------------------------------------------------------------------
# The prompt TEMPLATE
# -----------------------------------------------------------------------------
# {information} is a placeholder. PromptTemplate replaces it with real text
# when the chain runs. Everything else is sent to the model as-is.
# The numbered list is an "output contract": it tells the model exactly which
# three parts to return, so every run has the same shape.
PROFILE_TEMPLATE = """Here is a short biography:
{information}

Using only this text, give me:
1. A one-line headline about this person
2. A three-bullet summary a 12-year-old would understand
3. One question you would ask them in an interview
"""


# -----------------------------------------------------------------------------
# build_profile_chain()
# -----------------------------------------------------------------------------
# Creates the chain  PromptTemplate | ChatGoogleGenerativeAI.
#
#   Input : model_name (str) e.g. "gemini-2.5-flash"
#   Output: a Runnable. Call it with chain.invoke({"information": "..."})
def build_profile_chain(model_name: str):
    # input_variables lists every {placeholder} the template expects.
    profile_prompt_template = PromptTemplate(
        input_variables=["information"], template=PROFILE_TEMPLATE
    )

    # temperature=0 asks for the most deterministic answer the model can give.
    # NOTE: GPT-5.x reasoning models only support their default temperature;
    # LangChain quietly drops the setting for them, so this line is safe for
    # every model (and it does matter for local models in Script 03).
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        vertexai=True
    )

    # The pipe operator builds a "RunnableSequence": prompt first, then model.
    return profile_prompt_template | llm


# ↑ Example:
#   chain = build_profile_chain("gpt-5.6-sol")
#   chain.invoke({"information": "Sachin Tendulkar is a former Indian cricketer ..."})
#   → AIMessage(content=..., response_metadata={"model_name": "gpt-5.6-sol", ...})


def main() -> None:
    print("Hello from Future with Shashwat — AI Harness!")
    print(f"Model: {GEMINI_MODEL}\n")

    chain = build_profile_chain(GEMINI_MODEL)

    # invoke() runs the whole chain once. The dict keys must match the
    # template's placeholders ({information}).
    response = chain.invoke(input={"information": PERSON_INFORMATION})

    # `response` is an AIMessage (see Script 04 for everything inside it).
    # `.text` returns the answer as a plain string. We use `.text` instead of
    # `.content` because newer OpenAI models reply through the Responses API,
    # where `.content` may be a LIST of content blocks rather than a string.
    print(response.text)


if __name__ == "__main__":
    main()
