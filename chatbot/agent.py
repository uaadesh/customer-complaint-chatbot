# chatbot/agent.py

import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain_community.llms import OpenAI

from chatbot.config import use_openai, OPENAI_API_KEY
from chatbot.rag_retriever import build_retriever
from chatbot.tools import create_complaint_tool, get_complaint_tool


def get_local_llm(model_name="llama3.2:3b"):
    """
    Returns a locally running Ollama-compatible LLM client.
    """
    return ChatOllama(
        model=model_name,
        base_url="http://127.0.0.1:11434",
        temperature=0.3,
    )


def build_agent():
    """
    Builds the LangGraph-powered RAG agent with:
    - RAG for general knowledge-based answers
    - Tool usage for complaint creation and retrieval
    - Structured memory
    - JSON-based tool invocation
    """
    # System instruction prompt
    system_prompt = """
        You are a helpful and professional customer support assistant. Your job is to:

        1. File complaints by collecting user information:
        - name
        - 10-digit Indian phone number
        - valid email address
        - complaint details

        If any information is missing, ask politely for it. Once you have all details, output the following JSON to call the complaint creation tool:
        {
        "action": "create_complaint_tool",
        "input": "name, phone, email, complaint details"
        }

        2. Retrieve complaint details if the user gives a complaint ID, using this JSON format:
        {
        "action": "get_complaint_tool",
        "input": "complaint_id"
        }

        3. If no tool is needed, simply respond directly based on the knowledge base.

        Always format JSON properly on a new line. Never explain the JSON. Just return it clearly so the system can parse it.

        Example:
        User: I want to file a complaint.
        You: I'd be happy to help. May I know your name?

        Once all fields are collected:
        {
        "action": "create_complaint_tool",
        "input": "Deepak, 9876543210, deepak@example.com, My order was delayed"
        }
    """

    # Choose LLM backend
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY) if use_openai else get_local_llm()

    # Build RAG retriever QA system
    retriever = build_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Tool registry
    tools = {
        "GeneralQA": qa.run,
        "create_complaint_tool": create_complaint_tool,
        "get_complaint_tool": get_complaint_tool,
    }

    # Memory for conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # --- LangGraph Node: LLM Step ---
    def llm_step(input: dict):
        state = input["state"]
        user_input = state["user_input"]
        chat_history = state.get("chat_history", [])

        messages: list[BaseMessage] = [SystemMessage(content=system_prompt.strip())]
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                messages.append(HumanMessage(content=msg.content))
            elif isinstance(msg, AIMessage):
                messages.append(AIMessage(content=msg.content))
        messages.append(HumanMessage(content=user_input))

        response = llm.invoke(messages)
        llm_output = response.content.strip()

        return {
            "state": {
                **state,
                "llm_output": llm_output
            }
        }

    # --- LangGraph Node: Tool Step ---
    def tool_step(input: dict):
        state = input["state"]
        llm_output = state["llm_output"]
        tool_result = ""
        action = None

        try:
            # Extract JSON from LLM output
            json_start = llm_output.find("{")
            json_str = llm_output[json_start:] if json_start >= 0 else ""
            parsed = json.loads(json_str)

            action = parsed.get("action")
            tool_input = parsed.get("input")

            if action and action in tools:
                result = tools[action](tool_input)

                # Format human-friendly output based on action
                if action == "create_complaint_tool":
                    final_output = f"✅ Your complaint has been registered successfully. We’ll get back to you shortly. Here is your complaint ID: {result.split(':')[-1].strip()}"
                elif action == "get_complaint_tool":
                    if "Complaint ID" in result:
                        # Extract & format details nicely
                        lines = result.splitlines()
                        details = "\n".join(lines[1:])  # Skip first line
                        final_output = f"Here are the details of your complaint:\n\n{details}"
                    else:
                        final_output = "❌ Sorry, I couldn't find a complaint with that ID. Please double-check and try again."
                else:
                    final_output = result
            else:
                # JSON parsed but no valid action found
                final_output = state["llm_output"]  # Continue as normal response

        except json.JSONDecodeError:
            # Not a tool call — just proceed with LLM output as response
            final_output = state["llm_output"]

        except Exception as e:
            final_output = "⚠️ Something went wrong while processing your request. Please try again later."

        return {
            "state": {
                **state,
                "tool_result": tool_result,
                "final_output": final_output
            }
        }

    # Build LangGraph: llm → tool → END
    workflow = StateGraph(dict)
    workflow.add_node("llm", llm_step)
    workflow.add_node("tool", tool_step)
    workflow.add_edge("llm", "tool")
    workflow.add_edge("tool", END)
    workflow.set_entry_point("llm")
    graph = workflow.compile()

    # --- Agent class wrapper ---
    class LangGraphAgent:
        def __init__(self, graph, memory):
            self.graph = graph
            self.memory = memory

        def run(self, user_input: str) -> str:
            """
            Executes one turn of conversation with the agent.
            Returns the final agent reply string.
            """
            chat_history = getattr(self.memory.chat_memory, "messages", [])
            result = self.graph.invoke({
                "state": {
                    "user_input": user_input,
                    "chat_history": chat_history
                }
            })

            final_output = result["state"]["final_output"]

            # Update memory
            self.memory.chat_memory.messages.append(HumanMessage(content=user_input))
            self.memory.chat_memory.messages.append(AIMessage(content=final_output))

            return final_output

    return LangGraphAgent(graph, memory)
