#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 08:15:11 2024

@author: E. A. León-Gómez
"""


import pandas as pd
import warnings, json 

import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_ollama import ChatOllama
from langchain.llms import Ollama
from langchain_openai import ChatOpenAI

from Tutorials.keys import config 

warnings.filterwarnings("ignore")

# ==================================================================
# Load variables 
# ==================================================================

# Verificar si el archivo CSV existe
#file_path = "./data/train.csv"
file_path = "./data/synthetic-dataSet.csv"
df = pd.read_csv(file_path, index_col="timestamp")
#df = pd.read_csv(file_path)

# Crear el modelo de lenguaje
LLM_OpenAI = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=None, timeout=None, max_retries=2, api_key=config.api_key)
LLM_Llama  = Ollama(model="llama3.2:1b")

AVAILABLE_MODELS = ["llama3.2:1b", "gpt-3.5-turbo"]
DEFAULT_MODEL = "llama3.2:1b"

# ==================================================================
# PROMPT
# ==================================================================

prompt = """ Eres un asistente útil y siempre responderas en español. 

Responderas de la siguiente manera

- Para una pregunta sencilla que no necesita un gráfico ni una tabla, su respuesta debería ser:
{"answer": "Su respuesta va aquí"}

Por ejemplo:
{"answer": "El producto con el mayor número de pedidos es '15143Exfo'"}

- Si la consulta requiere una tabla, formatee su respuesta de esta manera:
{"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

- Para un gráfico de barras, responda de esta manera:
{"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

"""

# ==================================================================
# Functions
# ==================================================================

def create_sidebar() -> None:
    """
    Create and populate the sidebar with model settings and controls.

    This function sets up the sidebar in the Streamlit app, allowing users to adjust
    various parameters for the chat model, including the system prompt, model selection,
    and hyperparameters like temperature and top_p.
    """
    with st.sidebar:
        st.header("Configuración de inferencia")

        st.session_state.model = st.selectbox(
            "Modelos",
            AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(DEFAULT_MODEL),
            help="Seleccionar el modelo a usar.",
        )
        

def decode_response(response: str) -> dict:
    return json.loads(response)        

def write_answer(response_dict: dict):
    
    if "answer" in response_dict:
        with st.chat_message("assistant"):
            st.markdown(response_dict["answer"])

    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)
    
    if "bar" in response_dict:
        data = response_dict["bar"]
        try:
            # Crear el DataFrame con las columnas y los datos
            df = pd.DataFrame({
                "Feature": data["columns"],
                "Value": data["data"]
            })

            # Mostrar el DataFrame en Streamlit
            st.write("### Datos Procesados:")
            st.dataframe(df)

            # Dibujar el gráfico de barras
            st.write("### Gráfico de Barras:")
            st.bar_chart(df.set_index("Feature"))
        except ValueError as e:
            st.error(f"No se pudo crear el DataFrame a partir de los datos. Error: {e}")

# ==================================================================
# CODE
# ==================================================================

def main() -> None:

    # streamlit web app configuration
    st.set_page_config(page_title="Turing Chat", page_icon="💬", layout="centered" )
    st.title("🤖 Turing ")
    create_sidebar()

    # Initialize chat history in streamlit session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # input field for user's message
    user_prompt = st.chat_input("Pregúntale a Turing...")

    # ------------------------------------------------------------
    

    if user_prompt:
        # add user's message to chat history and display it
        st.chat_message("user").markdown(user_prompt)
        st.session_state.chat_history.append({"role":"user","content": user_prompt})

        agent = create_pandas_dataframe_agent(LLM_OpenAI,
                                              df,
                                              verbose=True,
                                              agent_type="openai-functions",
                                              allow_dangerous_code=True,
                                              include_df_in_prompt=True
                                              )

        messages = [
            {"role":"system", "content": prompt},
            *st.session_state.chat_history
        ]

        response = agent.invoke(messages)["output"]

        print("\n\t Pregunta ... \n {messages} \n\t".format(messages=messages))
        print("\n\t Respuesta ... \n {response} \n\t".format(response=response))

        # Decode the response.
        decoded_response = decode_response(response)

        print(decoded_response)

        # Write the response to the Streamlit app.
        final_response = write_answer(decoded_response)

        st.session_state.chat_history.append({"role":"assistant", "content": response})


    if st.button("Clear Chat History"):
        st.session_state.chat_history.clear()
        st.rerun()



if __name__ == "__main__":
    main()
