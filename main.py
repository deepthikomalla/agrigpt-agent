import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Annotated, Optional, TypedDict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from knowledge import PESTS_KNOWLEDGE_BASE, SCHEMES_KNOWLEDGE_BASE

load_dotenv()

def get_api_key() -> Optional[str]:
    env_path = Path('.') / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

app = FastAPI()

# 1. State Definition
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 2. Tool Logic
def simulate_pests(query: str):
    """Provides information on crop pests, diseases, and treatments."""
    return str(PESTS_KNOWLEDGE_BASE)

def government_schemes(query: str):
    """Provides details on government subsidies and schemes."""
    return str(SCHEMES_KNOWLEDGE_BASE)

pest_tool = StructuredTool.from_function(func=simulate_pests, name="simulate_pests", description="Pests info")
scheme_tool = StructuredTool.from_function(func=government_schemes, name="government_schemes", description="Schemes info")
all_tools = [pest_tool, scheme_tool]

# 3. Agent Builder
def build_agent(api_key: str):
    # Initialize the Gemini model via LangChain
    # Use a supported Gemini model name for the current API version
    model_name = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key,max_retries=3,
        temperature=0)
    llm_with_tools = llm.bind_tools(all_tools)

    def agent_node(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def tool_node(state: State):
        last_msg = state["messages"][-1]
        tool_outputs = []
        for tool_call in last_msg.tool_calls:
            # Map tool name string to actual execution
            if tool_call["name"] == "simulate_pests":
                result = simulate_pests(tool_call["args"])
            else:
                result = government_schemes(tool_call["args"])
            
            tool_outputs.append(ToolMessage(
                content=str(result), 
                tool_call_id=tool_call["id"], 
                name=tool_call["name"]
            ))
        return {"messages": tool_outputs}

    def should_continue(state: State):
        last = state["messages"][-1]
        return "tools" if (hasattr(last, "tool_calls") and last.tool_calls) else END

    workflow = StateGraph(State)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    return workflow.compile()

# 4. API Endpoints
class ChatRequest(BaseModel):
    chatId: str
    phone_number: str
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    api_key = get_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail="API Key not configured on server")
    agent = build_agent(api_key)
    system_msg = SystemMessage(content="You are an agri-expert. Use tools for agri-queries. If unrelated, say you only handle agriculture and DO NOT call tools.")

    try:
        result = agent.invoke({"messages": [system_msg, HumanMessage(content=request.message)]})
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Gemini API Error: {str(e)}"
        )

    # Capture sources correctly
    sources = [msg.name for msg in result["messages"] if isinstance(msg, ToolMessage)]

    return {
        "chatId": request.chatId,
        "response": result["messages"][-1].content,
        "sources": list(set(sources))
    }

@app.get("/status")
async def status():
    api_key_present = bool(get_api_key())
    model_name = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    return {
        "api_key_present": api_key_present,
        "model": model_name,
    }

@app.post("/schemes")
async def get_schemes(query: str):
    return {"results": SCHEMES_KNOWLEDGE_BASE}

@app.post("/pests")
async def get_pests(query: str):
    return {"results": PESTS_KNOWLEDGE_BASE}

if __name__ == "__main__":
    import uvicorn
    print("AgriGPT Server is starting...")
    # Using "main:app" ensures uvicorn loads the app correctly and stays running
    uvicorn.run("main:app", host="127.0.0.1", port=8030, reload=True)