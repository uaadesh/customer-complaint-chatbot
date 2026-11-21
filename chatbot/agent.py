# chatbot/agent.py

import json
import re
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
        temperature=0,    # deterministic: the same question gets the same answer every run
        num_predict=512,  # cap reply length so a model stuck in a loop can't run for minutes
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
        You are a helpful and professional customer support assistant for VegaKart, an Indian online store. Your job is to:

        1. File complaints. You need all four of these from the customer:
        - full name
        - 10-digit Indian phone number
        - email address
        - a description of the problem

        Ask politely for whatever is missing. Use only what the customer has actually told you: never invent values, and never reuse anything from these instructions.
        Only when you have all four, including the description of the problem, output this JSON to call the complaint creation tool, replacing each <...> with the customer's own words:
        {
        "action": "create_complaint_tool",
        "input": "<name>, <phone>, <email>, <description of the problem>"
        }

        2. Retrieve complaint details if the customer gives a complaint ID, using this JSON format:
        {
        "action": "get_complaint_tool",
        "input": "<complaint ID>"
        }

        3. If no tool is needed, simply respond directly based on the knowledge base.

        When you output JSON, output only the JSON: no explanation and no code fences.

        Always reply in the same language and script the customer uses: English, Hindi (Devanagari) or Hinglish (Hindi in Roman script).

        Example:
        Customer: I want to file a complaint.
        You: I'd be happy to help. May I know your full name?
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
        # Thinking models (e.g. qwen3.5) may put their reasoning in <think>...</think>
        # before the answer. Drop it, including an unclosed <think> cut off by the
        # length cap, so only the actual answer is parsed and shown.
        llm_output = re.sub(r"<think>.*?(</think>|$)", "", response.content, flags=re.DOTALL).strip()

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
            # raw_decode reads the first complete JSON object and ignores anything
            # after it, e.g. a closing ``` fence or extra sentences from the model.
            parsed, _ = json.JSONDecoder().raw_decode(json_str)

            action = parsed.get("action")
            tool_input = parsed.get("input")

            if action and action in tools:
                result = tools[action](tool_input)

                # Format human-friendly output based on action
                if action == "create_complaint_tool":
                    # Only claim success if the tool actually returned an ID,
                    # e.g. "✅ Complaint created successfully (ID: 1DCE63BD)".
                    match = re.search(r"\(ID: ([A-Z0-9]+)\)", result)
                    if match:
                        final_output = f"✅ Your complaint has been registered successfully. We’ll get back to you shortly. Here is your complaint ID: {match.group(1)}"
                    else:
                        final_output = "❌ Sorry, I couldn't register your complaint. Please check your name, 10-digit phone number, email and complaint details, and try again."
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
