"""
Streamlit chat UI for AgentA1_v1. Run with: streamlit run ui.py
"""

import streamlit as st

import agent

st.set_page_config(page_title="A1 Tutor", page_icon="🎓")

st.title("A1 English Tutor")
st.caption(f"Model: {agent.OLLAMA_MODEL} · replies are validated against A1spec.json")

if "conversation_history" not in st.session_state:
    greeting = "Hello! How are you?"
    st.session_state.conversation_history = [{"role": "assistant", "content": greeting}]
    st.session_state.logger = agent.Logger()
    st.session_state.logger.log("greeting", text=greeting)

for message in st.session_state.conversation_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.conversation_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply, attempts, valid, reasons = agent.get_reply(
                    st.session_state.conversation_history,
                    user_input,
                    st.session_state.logger,
                )
            except RuntimeError as e:
                reply, attempts, valid, reasons = f"[ERROR] {e}", 0, True, []

        st.write(reply)
        if not valid:
            st.caption(f"⚠ Needed {attempts} attempt(s); still flagged: {', '.join(reasons)}")
        elif attempts > 1:
            st.caption(f"✓ Passed validation after {attempts} attempts")

    st.session_state.conversation_history.append({"role": "assistant", "content": reply})

    if len(st.session_state.conversation_history) > 12:
        st.session_state.conversation_history = st.session_state.conversation_history[-12:]
