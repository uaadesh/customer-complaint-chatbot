import sys
import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
torch.classes.__path__ = []

import streamlit as st
from chatbot.agent import build_agent
from langchain.schema import HumanMessage, AIMessage

st.set_page_config(page_title="Complaint Assistant", page_icon="🧾", layout="centered")

st.title("🧾 Complaint Chatbot Assistant")
st.markdown("This assistant can answer customer service questions, file complaints, and retrieve complaints using their ID.")

if "agent" not in st.session_state:
    try:
        st.session_state.agent = build_agent()
    except Exception as e:
        st.error(f"Agent initialization failed: {e}")
        st.stop()

user_input = st.chat_input("Ask something...")
if user_input:
    with st.spinner("Thinking..."):
        response = st.session_state.agent.run(user_input)

# # 🔁 Display chat history from agent memory
# chat_history = (
#     st.session_state.agent.memory.chat_memory.messages
#     if hasattr(st.session_state.agent, "memory")
#     else []
# )
# for msg in chat_history:
#     if isinstance(msg, HumanMessage):
#         with st.chat_message("user"):
#             st.markdown(msg.content)
#     elif isinstance(msg, AIMessage):
#         with st.chat_message("assistant"):
#             st.markdown(msg.content)


import re

# 🔁 Display chat history from agent memory
chat_history = (
    st.session_state.agent.memory.chat_memory.messages
    if hasattr(st.session_state.agent, "memory")
    else []
)

for idx, msg in enumerate(chat_history):
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)

    elif isinstance(msg, AIMessage):
        # Split response into visible and <think> part
        content = msg.content
        match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if match:
            visible = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            thought = match.group(1).strip()
        else:
            visible = content.strip()
            thought = None

        with st.chat_message("assistant"):
            st.markdown(visible)
            if thought:
                with st.expander("🧠 Show thought process"):
                    st.markdown(thought)
