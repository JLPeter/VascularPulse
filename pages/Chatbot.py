import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Chatbot")
st.title("Medical Questions? Ask the Chatbot.")
st.caption("Powered by Hugging Face Inference API using Qwen2.5. Please note that this model is an LLM and may produce "
           "factually incorrect or logically inconsistent responses. Always verify generated content for accuracy.")

#HuggingFace API
client = InferenceClient(token=st.secrets["HF_API_TOKEN"])

def get_chat_response(prompt: str) -> str:
    try:
        completion = client.chat.completions.create(
            model="jjzha/Qwen2.5-0.5B-Instruct-rt",
            messages=st.session_state.messages + [{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        return completion.choices[0].message["content"]
    except Exception as e:
        return f"API error: {e}. Please reload and try again."

#Session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hi there! How can I help you today?"})

#Display history of chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#Input box
if prompt := st.chat_input("Type your message here..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    #Get response from chatbot
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_chat_response(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})


