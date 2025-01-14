
import logging
from typing import Dict, Any

import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama

# Constants
PAGE_TITLE = "Llama 3.2 Chat"
PAGE_ICON = "🦙"
SYSTEM_PROMPT = "You are a friendly and informative AI chatbot conversing with a human."
DEFAULT_MODEL = "llama3.2:latest"
AVAILABLE_MODELS = ["llama3.2:1b", "llama3.2:latest"]

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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


def create_sidebar() -> None:
    """
    Create and populate the sidebar with model settings and controls.

    This function sets up the sidebar in the Streamlit app, allowing users to adjust
    various parameters for the chat model, including the system prompt, model selection,
    and hyperparameters like temperature and top_p.
    """
    with st.sidebar:
        st.header("Inference Settings")
        st.session_state.system_prompt = st.text_area(
            label="System",
            value=SYSTEM_PROMPT,
            help="Sets the context for the AI model interaction.",
        )

        st.session_state.model = st.selectbox(
            "Model",
            AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(DEFAULT_MODEL),
            help="Select the model to use.",
        )
        st.session_state.seed = st.number_input(
            "Seed",
            min_value=1,
            max_value=9_007_199_254_740_991,
            value=st.session_state.seed,
            help="Controls the randomness in text generation.",
        )
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.01,
            help="Controls the randomness in the output.",
        )
        st.session_state.top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.top_p,
            step=0.01,
            help="Controls the diversity of the model's responses.",
        )
        st.session_state.num_predict = st.slider(
            "Response Tokens",
            min_value=0,
            max_value=8192,
            value=st.session_state.num_predict,
            step=16,
            help="Sets the maximum number of tokens in the response.",
        )

        display_model_stats()


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

        with st.spinner("Thinking..."):
            config = {"configurable": {"session_id": "any"}}
            response = chain_with_history.invoke({"input": prompt}, config)
            logger.info({"input": prompt}, config)
            st.chat_message("ai").write(response.content)
            logger.info(response)
            update_sidebar_stats(response)


def main() -> None:
    """
    Main function to run the Streamlit chat application.

    This function sets up the page configuration, initializes the session state,
    creates the sidebar, and handles the chat interaction loop. It manages the
    chat history, processes user inputs, and displays AI responses.
    """
    setup_page_config()
    initialize_session_state()
    create_sidebar()

    chat_model = create_chat_model()
    chain = create_chat_chain(chat_model)

    msgs = StreamlitChatMessageHistory(key="special_app_key")
    if not msgs.messages:
        msgs.add_ai_message("How can I help you?")

    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda _: msgs,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    handle_user_input(chain_with_history)

    if st.button("Clear Chat History"):
        msgs.clear()
        st.rerun()


if __name__ == "__main__":
    main()
