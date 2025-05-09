from dotenv import load_dotenv
import os
from typing import Annotated, Literal
from langchain_openai import ChatOpenAI  # Correct import
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph import START, END 

load_dotenv()

# Initialize OpenAI o4-mini correctly
llm = ChatOpenAI(model="o4-mini")  # Direct initialization

class MessageClassifier(BaseModel):
    message_type: Literal["emotional","logical"] = Field(
        ...,
        descriptopn="classify the message requires an emotional or logical response."
    )

class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None

# This function classifies the last message in the conversation
def classify_message(state: State): 
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
           "role": "system",
           "content": """Classift the user message as either:
            - 'emotional': if it ask for emotional support, therapy, or empathy.
            - 'logical': if it ask for logical reasoning, advice, or information.
            """
        },
        {"role": "user", "content": last_message.content}
    ])
    return{"message_type":result.message_type}

 # This function routes the conversation based on the message type
def router(state: State):
    message_type= state.get("message_type","logical")
    if message_type == "emotional":
        return {"next":"therapist"}
    
    return {"next": "logical"}   

# This function handles the conversation with the therapist agent
def therapist_agent(state: State):
    last_message = state["messages"][-1]
    messages=[
        {"role": "system",
        "content": """You are a therapist. 
        You are here to provide emotional support, therapy, and empathy to the user.
        """
        },
        {
        "role": "user",
        "content": last_message.content
        }
    ]
    reply=llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

# This function handles the conversation with the logical agent
def logical_agent(state: State):
    last_message = state["messages"][-1]
    messages=[
        {"role": "system",
        "content": """You are a logical assistant. 
        You are here to provide clear, accurate, concise answers based on evidence to the user.
        """
        },
        {
        "role": "user",
        "content": last_message.content
        }
    ]
    reply=llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

# Define the state graph
graph_builder = StateGraph(State)    
graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {"therapist": "therapist", "logical": "logical"}
)

graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)

# Add edges for the therapist and logical agents
graph = graph_builder.compile()

def run_chatbot():
    # Initialize the state with an empty message list
    state = {"messages": [], "message_type": None}
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("bye")
            break
        
        # Add the user's message to the state
        state["messages"]=state.get("messages", []) + [{"role": "user", "content": user_input}]
        
        # Run the state graph
        state = graph.invoke(state)
        
        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assiant: {last_message.content}")

if __name__ == "__main__":
    run_chatbot()
# This code is a simple chatbot that uses LangChain and OpenAI's GPT-3.5-turbo model to classify user messages and respond accordingly.
# It classifies messages as either "emotional" or "logical" and routes them to the appropriate agent for response.
# The chatbot continues to run until the user types "exit".