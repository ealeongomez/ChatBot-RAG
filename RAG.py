#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 5 14:34:00 2024

@author: E. A. León-Gómez
"""

import logging
import streamlit as st
from typing import Union
from typing import Dict, Any

from langchain import PromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

from Tutorials.keys import config

from langsmith import traceable
from dotenv import load_dotenv

from get_embedding_function import get_embedding_function

logging.basicConfig(
    level=logging.INFO,
    format=" %(asctime)s - %(name)s - %(levelname)s - %(message)s \n\n",  
)

logger = logging.getLogger(__name__)

load_dotenv(dotenv_path="./Tutorials/.env", override=True)

# ================================================================================================
# Constants
# ================================================================================================
PAGE_TITLE = "Chatbot RAG"
PAGE_ICON = "🦙"
DEFAULT_MODEL = "llama3.2:1b"
DEFAULT_DB = "Chroma"

AVAILABLE_MODELS = ["llama3.2:1b", "gpt-3.5-turbo"]
AVAILABLE_DB = ["Chroma", "Elastic Search"]

temperature = 0.5

# ------------------------------------------------------------------------------------------------

PROMPT = """Eres un Modelo de Lenguaje Natural, llamado Clico."""

PROMPT_TEMPLATE = """Eres un Modelo de Lenguaje Natural, llamado Clico.
Tu misión es ayudar a los compañeros de trabajo a mejorar su experiencia y ayudarles. Utiliza los siguientes fragmentos de contexto para responder a la pregunta al final.

{context}

---

Responda la pregunta según el contexto anterior: {question}
"""

# ================================================================================================
# RAG
# ================================================================================================

CHROMA_PATH = './Chroma-Constitucion'

embedding_function = get_embedding_function()

db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# ================================================================================================
# Initilization 
# ================================================================================================

def create_sidebar() -> None:
    with st.sidebar:
        st.session_state.model = st.selectbox(
            "Model",
            AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(DEFAULT_MODEL),
            help="Seleccione el modelo a utilizar",
        )

        st.session_state.DB = st.selectbox(
            "Data base",
            AVAILABLE_DB,
            index=AVAILABLE_DB.index(DEFAULT_DB),
            help="Seleccione la base de datos",
        )

        #display_model_stats()

# ------------------------------------------------------------------------------------------------

def display_model_stats() -> None:
    st.markdown("---")
    st.text(
        f"""Stats:
            - model: {st.session_state.model}
        """
    )

# ================================================================================================ 
# Model 
# ================================================================================================ 

def create_chat_model() -> object:
    """
    Create and configure a chat model instance based on the current session state.

    Returns:
        object: A configured chat model instance for use in the chat application.
    """
    if st.session_state.model == "llama3.2:1b":
        return ChatOllama(model=st.session_state.model, temperature=temperature)
    elif st.session_state.model == "gpt-3.5-turbo":
        return ChatOpenAI(
            model=st.session_state.model,
            temperature=temperature,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=config.api_key
        )
    else:
        raise ValueError(f"Unsupported model: {st.session_state.model}")

# ------------------------------------------------------------------------------------------------

def create_chat_chain(chat_model: Union[ChatOllama, ChatOpenAI]):
    """
    Create a chat chain using the provided chat model and system prompt.

    Args:
        chat_model (Union[ChatOllama, ChatOpenAI]): The chat model instance to use in the chain.

    Returns:
        A chat chain combining the system prompt, chat history, and user input.
    """
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    # Prompt logging -------------------------------------------
    logger.info(f"\n\t")
    logger.info(prompt.messages[0])

    return prompt | chat_model


# ------------------------------------------------------------------------------------------------

def setup_page_config() -> None:
    st.set_page_config(page_title="Turing Chat", page_icon="💬", layout="centered" )
    st.title("🤖 Turing ")
    st.markdown(
        """
        <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("##### Chat")

# ------------------------------------------------------------------------------------------------
 
#  DATA BASES   
#     - ChromaDB
#             # Search in database ----------------------------------------------------------------
#             results = db.similarity_search_with_score(prompt, k=1)
#             if len(results) == 0:
#                 print(f"Unable to find matching results.")

#             context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

#             prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

#             # Without history
#             prompt_2 = prompt_template.format(context=context, question=prompt)

#             logger.info("----------------------------------------------------------------")

#             logger.info(prompt_template)

#             logger.info(prompt_2)

#             #logger.info(response.content)
#             # -----------------------------------------------------------------------------------
 

# ------------------------------------------------------------------------------------------------

def handle_user_input(chain_with_history: RunnableWithMessageHistory) -> None:
    """
    Handle user input and generate AI response.

    This function processes the user's input, sends it to the AI model,
    and displays the response in the Streamlit chat interface.

    Args:
        chain_with_history (RunnableWithMessageHistory): The chat chain with history management.
    """
    if prompt := st.chat_input("Pregúntale a Turing..."):
        st.chat_message("human").write(prompt)

        with st.spinner("Pensando..."):
            config = {"configurable": {"session_id": "any"}}
            response = chain_with_history.invoke({"input": prompt}, config)
            
            #logger.info({"input": prompt}, config)
            logger.info(prompt)

            st.chat_message("ai").write(response.content)
            logger.info(response)


# ================================================================================================ 
# ================================================================================================

def main() -> None:

    setup_page_config()

    # Parameter tuning ---------------------------------------------------------------------------
    create_sidebar()

    chat_model = create_chat_model()
    chain = create_chat_chain(chat_model)
    
    #logger.info(f"{chain}")

    msgs = StreamlitChatMessageHistory(key="special_app_key")
    if not msgs.messages:
        msgs.add_ai_message("Hola me llamo Clico ¿Cómo puedo ayudarte?")

    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda _: msgs,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)
        logger.info(msg.content)

    handle_user_input(chain_with_history)

    if st.button("Clear Chat History"):
        msgs.clear()
        st.rerun()


if __name__ == "__main__":
    main()