"""
Streamlit chat UI for AgentA1_v1 — OpenAI only, no Ollama.
Run with: streamlit run ui.py
"""

import streamlit as st

import agent

st.set_page_config(page_title="A1 Tutor", page_icon="🎓")

st.title("A1 English Tutor")

# Hard-require OpenAI. agent.py resolves OPENAI_API_KEY from Streamlit
# secrets / env var / .env, and OPENAI_INIT_ERROR explains why it's missing
# or why no candidate model worked, without ui.py needing to know details.
if not agent.OPENAI_API_KEY or not agent.OPENAI_MODEL:
    st.error(
        "**OpenAI is not configured.**\n\n"
        f"{agent.OPENAI_INIT_ERROR}\n\n"
        "Set `OPENAI_API_KEY` in this app's Secrets (Settings → Secrets on "
        "Streamlit Cloud) or in a local `.env` file, then reload the app."
    )
    st.stop()

st.caption(f"Model: {agent.OPENAI_MODEL} (OpenAI) · replies are validated against A1spec.json")

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
                    backend="openai",
                )
            except RuntimeError as e:
                reply = None
                st.error(f"Could not get a reply from OpenAI: {e}")

        if reply is not None:
            st.write(reply)
            if not valid:
                st.caption(f"⚠ Needed {attempts} attempt(s); still flagged: {', '.join(reasons)}")
            elif attempts > 1:
                st.caption(f"✓ Passed validation after {attempts} attempts")

    # Only remember successful turns — a failed call leaves no assistant
    # message in history, so the user's message is still there to retry against.
    if reply is not None:
        st.session_state.conversation_history.append({"role": "assistant", "content": reply})

    if len(st.session_state.conversation_history) > 12:
        st.session_state.conversation_history = st.session_state.conversation_history[-12:]
