from typing import Annotated, List, Dict
from datetime import datetime
import json

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)



@tool()
def validate_user(user_id: int, addresses: List[str]) -> bool:
    """Validate user using historical addresses.

    Args:
        user_id (int): the user ID.
        addresses (List[str]): Previous addresses as a list of strings.
    """
    return True

@tool()
def post_tweet(content: str) -> Dict[str, str]:
    """Post a simulated tweet by writing content and time to a JSON file.

    Args:
        content (str): The content of the tweet.
    """
    tweet = {
        "content": content,
        "time": datetime.now().isoformat()
    }

    try:
        with open("tweets.json", "r+") as file:
            data = json.load(file)
            data.append(tweet)
            file.seek(0)
            json.dump(data, file, indent=4)
    except FileNotFoundError:
        with open("tweets.json", "w") as file:
            json.dump([tweet], file, indent=4)

    return tweet

tools = [post_tweet]  # Update the tools list to include the correct function
llm = ChatOllama(model="nemotron-mini:4b-instruct-q4_K_M")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

#################################################

config = {"configurable": {"thread_id": "1"}}

user_input = "You just watched a good episode of your favourite anime. You decide to tweet about it."

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [{"role": "system", "content": "You are roleplaying as 19 year old Tyler, a angsty weeb with a passion in cooking."},
        {"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()