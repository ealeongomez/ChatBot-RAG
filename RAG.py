"""
Ollama-Streamlit-LangChain-Chat-App
Streamlit app for chatting with Meta Llama 3.2 using Ollama and LangChain
Author: Gary A. Stafford
Date: 2024-09-26
"""

import logging
from typing import Dict, Any

import streamlit as st
from langchain import PromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma

from get_embedding_function import get_embedding_function

# ======================================================================
# Constants
# ======================================================================
PAGE_TITLE = "Chatbot RAG"
PAGE_ICON = "🦙"
SYSTEM_PROMPT = "Eres un Modelo de Lenguaje Natural"
DEFAULT_MODEL = "llama3.2:1b"
AVAILABLE_MODELS = ["llama3.2:1b", "llama3.2:latest"]

PROMPT_TEMPLATE = """Eres un Modelo de Lenguaje Natural, llamado Clico.
Tu misión es ayudar a los compañeros de trabajo a mejorar su experiencia y ayudarles. Utiliza los siguientes fragmentos de contexto para responder a la pregunta al final.

{context}

---

Responda la pregunta según el contexto anterior: {question}
"""

# ======================================================================
# Configure logging 
# ======================================================================

logging.basicConfig(
    level=logging.INFO,
    format=" %(asctime)s - %(name)s - %(levelname)s - %(message)s \n\n",  
)

logger = logging.getLogger(__name__)

# ================================================================================================
# RAG
# ================================================================================================

CHROMA_PATH = './Chroma-Constitucion'

embedding_function = get_embedding_function()

db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# ======================================================================
# Initilization 
# ======================================================================

def initialize_session_state() -> None:
    """
    Initialize the Streamlit session state with default values.

    This function sets up default values for various parameters used in the chat application,
    including the model, token counts, and model hyperparameters. It only initializes
    values that haven't been set in the session state yet.
    """
    defaults: Dict[str, Any] = {
        "model": DEFAULT_MODEL,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_duration": 0,
        "num_predict": 2048,
        "seed": 4_503_599_627_370_496,  # Midpoint of the valid range
        "temperature": 0.5,
        "top_p": 0.9,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ====================================================================== 
# ======================================================================

def create_sidebar() -> None:
    """
    Create and populate the sidebar with model settings and controls.

    This function sets up the sidebar in the Streamlit app, allowing users to adjust
    various parameters for the chat model, including the system prompt, model selection,
    and hyperparameters like temperature and top_p.
    """
    with st.sidebar:
        st.header("Configuración de inferencia")
        st.session_state.system_prompt = st.text_area(
            label="System",
            value=SYSTEM_PROMPT,
            help="Establece el contexto para la interacción del modelo de IA..",
        )

        st.session_state.model = st.selectbox(
            "Model",
            AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(DEFAULT_MODEL),
            help="Seleccione el modelo a utilizar",
        )
        st.session_state.seed = st.number_input(
            "Seed",
            min_value=1,
            max_value=9_007_199_254_740_991,
            value=st.session_state.seed,
            help="Controla la aleatoriedad en la generación de texto",
        )
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.01,
            help="Controla la aleatoriedad en la salida",
        )
        st.session_state.top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.top_p,
            step=0.01,
            help="Controla la diversidad de las respuestas del modelo",
        )
        st.session_state.num_predict = st.slider(
            "Response Tokens",
            min_value=0,
            max_value=8192,
            value=st.session_state.num_predict,
            step=16,
            help="Establece el número máximo de tokens en la respuesta",
        )

        #display_model_stats()

# ====================================================================== 
# ======================================================================

def display_model_stats() -> None:
    """
    Display current model statistics in the sidebar.

    This function shows the current values of model parameters and settings
    in the Streamlit sidebar.
    """
    st.markdown("---")
    st.text(
        f"""Stats:
- model: {st.session_state.model}
- seed: {st.session_state.seed}
- temperature: {st.session_state.temperature}
- top_p: {st.session_state.top_p}
- num_predict: {st.session_state.num_predict}
        """
    )

# ======================================================================
# Model 
# ======================================================================

def create_chat_model() -> ChatOllama:
    """
    Create and configure a ChatOllama instance based on the current session state.

    Returns:
        ChatOllama: A configured ChatOllama instance for use in the chat application.
    """
    return ChatOllama(
        model=st.session_state.model,
        seed=st.session_state.seed,
        temperature=st.session_state.temperature,
        top_p=st.session_state.top_p,
        num_predict=st.session_state.num_predict,
    )


def create_chat_chain(chat_model: ChatOllama):
    """
    Create a chat chain using the provided chat model and system prompt.

    Args:
        chat_model (ChatOllama): The ChatOllama instance to use in the chain.

    Returns:
        A chat chain combining the system prompt, chat history, and user input.
    """
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", st.session_state.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    # Prompt ----------------------------------------------------
    logger.info(f"\n\t")
    logger.info(prompt.messages[0])
    #logger.info(f"prompt: {prompt.messages[0].format().content}")
    #logger.info(f"{input}")

    return prompt | chat_model


def update_sidebar_stats(response: Any) -> None:
    """
    Update the sidebar with statistics from the latest model response.

    This function calculates and updates various statistics in the session state,
    including token counts, duration, and tokens per second.

    Args:
        response (Any): The response object from the chat model containing metadata.
    """
    total_duration = response.response_metadata["total_duration"] / 1e9
    st.session_state.total_duration = f"{total_duration:.2f} s"
    st.session_state.input_tokens = response.usage_metadata["input_tokens"]
    st.session_state.output_tokens = response.usage_metadata["output_tokens"]
    st.session_state.total_tokens = response.usage_metadata["total_tokens"]
    token_per_second = (
        response.response_metadata["eval_count"]
        / response.response_metadata["eval_duration"]
    ) * 1e9
    st.session_state.token_per_second = f"{token_per_second:.2f} tokens/s"

    with st.sidebar:
        st.text(
            f"""
- input_tokens: {st.session_state.input_tokens}
- output_tokens: {st.session_state.output_tokens}
- total_tokens: {st.session_state.total_tokens}
- total_duration: {st.session_state.total_duration}
- token_per_second: {st.session_state.token_per_second}
        """
        )


def setup_page_config() -> None:
    """
    Set up the Streamlit page configuration.

    This function configures the Streamlit page, including the title, icon,
    layout, and custom CSS to hide certain elements.
    """
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
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
    st.title(f"{PAGE_TITLE} {PAGE_ICON}")
    st.markdown("##### Chat")


def handle_user_input(chain_with_history: RunnableWithMessageHistory) -> None:
    """
    Handle user input and generate AI response.

    This function processes the user's input, sends it to the AI model,
    and displays the response in the Streamlit chat interface.

    Args:
        chain_with_history (RunnableWithMessageHistory): The chat chain with history management.
    """
    if prompt := st.chat_input("Type your message here..."):
        st.chat_message("human").write(prompt)

        with st.spinner("Pensando..."):
            config = {"configurable": {"session_id": "any"}}
            response = chain_with_history.invoke({"input": prompt}, config)
            
            #logger.info({"input": prompt}, config)
            logger.info(prompt)

            # Search in database ----------------------------------------------------------------
            results = db.similarity_search_with_score(prompt, k=1)
            if len(results) == 0:
                print(f"Unable to find matching results.")

            context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

            # Without history
            prompt_2 = prompt_template.format(context=context, question=prompt)


            logger.info("----------------------------------------------------------------")

            logger.info(prompt_2)

            #logger.info(response.content)
            # -----------------------------------------------------------------------------------

            st.chat_message("ai").write(response.content)
            logger.info(response)
            update_sidebar_stats(response)


# ================================================================================================ 
# ================================================================================================

def main() -> None:
    """
    Main function to run the Streamlit chat application.

    This function sets up the page configuration, initializes the session state,
    creates the sidebar, and handles the chat interaction loop. It manages the
    chat history, processes user inputs, and displays AI responses.
    """

    #setup_page_config()

    # Parameter tuning ---------------------------------------------------------------------------
    initialize_session_state()
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
