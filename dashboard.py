import pandas as pd

import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_ollama import ChatOllama
from langchain.llms import Ollama
from langchain_openai import ChatOpenAI

from Tutorials.keys import config 

# ==================================================================
# Load variables 
# ==================================================================

# Verificar si el archivo CSV existe
file_path = "./data/train.csv"
df = pd.read_csv(file_path)

# Crear el modelo de lenguaje
LLM_OpenAI = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=None, timeout=None, max_retries=2, api_key=config.api_key)
LLM_Llama  = Ollama(model="llama3.2:1b")

# ==================================================================
# CODE
# ==================================================================

def main() -> None:

    # streamlit web app configuration
    st.set_page_config(page_title="Turing Chat", page_icon="💬", layout="centered" )
    st.title("🤖 Turing ")

    # Initialize chat history in streamlit session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # input field for user's message
    user_prompt = st.chat_input("Ask LLM...")

    if user_prompt:
        # add user's message to chat history and display it
        st.chat_message("user").markdown(user_prompt)
        st.session_state.chat_history.append({"role":"user","content": user_prompt})

        pandas_df_agent = create_pandas_dataframe_agent(LLM_OpenAI, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS, allow_dangerous_code=True)

        pandas_df_agent = create_pandas_dataframe_agent(LLM_OpenAI,
                                                        df,
                                                        verbose=True,
                                                        agent_type="openai-functions",
                                                        allow_dangerous_code=True,
                                                        include_df_in_prompt=True
                                                        )

        messages = [
            {"role":"system", "content": "Eres un asistente útil"},
            *st.session_state.chat_history
        ]

        response = pandas_df_agent.invoke(messages)

        """
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = pandas_df_agent.run(messages, callbacks=[st_cb])
            st.write(response)
        """

        assistant_response = response["output"]

        st.session_state.chat_history.append({"role":"assistant", "content": assistant_response})

        # display LLM response
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

    if st.button("Clear Chat History"):
        st.session_state.chat_history.clear()
        st.rerun()



if __name__ == "__main__":
    main()
