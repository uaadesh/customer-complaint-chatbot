# chatbot/agent.py

import re
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langchain.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama

from chatbot.config import use_openai, LLM_THINKING
from chatbot.tools import create_complaint_tool, get_complaint_tool

FALLBACK_REPLY = "Sorry, I didn't quite get that. Could you please rephrase?"


def get_local_llm(model_name="gemma4:e2b"):
    """
    Returns a locally running Ollama-compatible LLM client.

    gemma4:e2b was chosen in the v0 model benchmark (see evals/results/v0-summary.md).
    """
    return ChatOllama(
        model=model_name,
        base_url="http://127.0.0.1:11434",
        temperature=0,    # deterministic: the same question gets the same answer every run
        num_predict=512,  # cap reply length so a model stuck in a loop can't run for minutes
        # Put all model layers on the GPU. By default Ollama keeps some on the CPU
        # because it reserves GPU memory for gemma's image/audio parts, which this
        # text-only bot never uses. All on GPU = ~30% faster replies on a 4 GB GPU.
        num_gpu=99,
        # gemma4 "thinks" (writes hidden reasoning) before every reply unless told not
        # to. Thinking text is kept out of the reply either way; see LLM_THINKING in config.
        reasoning=LLM_THINKING,
    )


def build_agent():
    """
    Builds the LangGraph-powered agent with:
    - Native tool calling for complaint creation and lookup
    - Structured memory
    """
    # System instruction prompt. The tools' names, arguments and descriptions are sent
    # to the model separately, as tool schemas (see chatbot/tools.py).
    system_prompt = """
        You are a helpful and professional customer support assistant for VegaKart, an Indian online store. Your job is to:

        1. File complaints with create_complaint_tool. You need all four of these from the customer:
        - full name
        - 10-digit Indian phone number
        - email address
        - a description of the problem

        Ask politely for whatever is missing. Use only what the customer has actually told you: never invent values, and never reuse anything from these instructions.
        Only call the tool when you have all four, including the description of the problem.

        2. Retrieve complaint details with get_complaint_tool if the customer gives a complaint ID.

        3. If no tool is needed, simply respond directly based on the knowledge base.

        Always reply in the same language and script the customer uses: English, Hindi (Devanagari) or Hinglish (Hindi in Roman script).

        Example:
        Customer: I want to file a complaint.
        You: I'd be happy to help. May I know your full name?
    """

    if use_openai:
        # The old OpenAI completion LLM can't do native tool calling. A chat model
        # with tool support (e.g. a free-tier API) can be added here later.
        raise NotImplementedError("use_openai is not supported since the switch to native tool calling")

    # Tool registry: name -> LangChain tool. bind_tools sends their schemas to the model.
    tools = {t.name: t for t in (create_complaint_tool, get_complaint_tool)}
    llm = get_local_llm().bind_tools(list(tools.values()))

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
        # Safety net for models that put reasoning inside the reply as <think>...</think>
        # (including an unclosed <think> cut off by the length cap). Ollama normally
        # returns thinking separately, so this rarely does anything.
        llm_output = re.sub(r"<think>.*?(</think>|$)", "", response.content, flags=re.DOTALL).strip()

        return {
            "state": {
                **state,
                "llm_output": llm_output,
                "tool_calls": response.tool_calls,  # structured calls: [{"name", "args", "id"}]
            }
        }

    # --- LangGraph Node: Tool Step ---
    def tool_step(input: dict):
        state = input["state"]
        llm_output = state["llm_output"]
        tool_calls = state.get("tool_calls") or []
        tool_result = ""

        if not tool_calls:
            # No tool needed: the model's text is the reply.
            final_output = llm_output or FALLBACK_REPLY
        elif tool_calls[0]["name"] not in tools:
            # The model called a tool that doesn't exist; don't run anything.
            final_output = llm_output or FALLBACK_REPLY
        else:
            call = tool_calls[0]  # one tool per turn
            action = call["name"]
            # Small models sometimes send a phone number as a number, not text.
            args = {k: str(v) for k, v in (call.get("args") or {}).items()}
            try:
                tool_result = tools[action].invoke(args)
            except Exception as e:  # e.g. a required argument is missing
                tool_result = f"❌ Invalid tool arguments: {e}"

            # Format a human-friendly reply from the tool result.
            if action == "create_complaint_tool":
                # Only claim success if the tool actually returned an ID,
                # e.g. "✅ Complaint created successfully (ID: 1DCE63BD)".
                match = re.search(r"\(ID: ([A-Z0-9]+)\)", tool_result)
                if match:
                    final_output = f"✅ Your complaint has been registered successfully. We’ll get back to you shortly. Here is your complaint ID: {match.group(1)}"
                else:
                    final_output = "❌ Sorry, I couldn't register your complaint. Please check your name, 10-digit phone number, email and complaint details, and try again."
            else:  # get_complaint_tool
                if "Complaint ID" in tool_result:
                    details = "\n".join(tool_result.splitlines()[1:])  # skip the ID line
                    final_output = f"Here are the details of your complaint:\n\n{details}"
                else:
                    final_output = "❌ Sorry, I couldn't find a complaint with that ID. Please double-check and try again."

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
