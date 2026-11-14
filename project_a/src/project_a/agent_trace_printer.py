# -----------------------------------------------------------------------------
# _one_line()
# -----------------------------------------------------------------------------
# Squashes any text to a single line and cuts it to `limit` characters, so long
# search results don't flood the terminal.
#
#   _one_line("a\nb\nc", 10)   → "a b c"
#   _one_line("x" * 500, 10)   → "xxxxxxxxxx … (+490 chars)"
def _one_line(text: str, limit: int = 160) -> str:
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return f"{flat[:limit]} … (+{len(flat) - limit} chars)"


# -----------------------------------------------------------------------------
# print_agent_trace()
# -----------------------------------------------------------------------------
#   Input : result (dict) — the value returned by agent.invoke(...)
#           It always has a "messages" key; with response_format=... it also
#           has a "structured_response" key (see Script 04).
#   Output: None (prints to the terminal)
def print_agent_trace(result: dict) -> None:
    messages = result["messages"]
    print(f"\nThe agent produced {len(messages)} messages:")
    for index, message in enumerate(messages):
        kind = type(message).__name__
        # AIMessages that call tools have an empty text and a tool_calls list.
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            for call in tool_calls:
                print(
                    f"  [{index}] {kind:<13}: 🛠  tool call → {call['name']}({_one_line(call['args'], 120)})"
                )
        elif kind == "ToolMessage":
            print(f"  [{index}] {kind:<13}: ({message.name}) {_one_line(message.text)}")
        else:
            print(f"  [{index}] {kind:<13}: {_one_line(message.text)}")

    # The final answer is simply the text of the LAST message.
    print("\nFINAL ANSWER:")
    print(messages[-1].text)


# ↑ Example:
#   result = agent.invoke({"messages": [HumanMessage(content="What's the weather like in Bengaluru right now?")]})
#   print_agent_trace(result)
#   → prints the 4-line trail shown in the module docstring, then the final answer.
