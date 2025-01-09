import os, warnings, re, unicodedata
import pandas as pd

from Tutorials.keys import config  
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Desactivar advertencias específicas
warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================================
# Variables
# ==============================================================================================

# Configuración del agente de OpenAI
LLM = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_retries=2, api_key=config.api_key)
#LLM  = Ollama(model="me/llama3.1-python")
#LLM  = Ollama(model="llama3.2")

df = pd.read_excel("./data/tableElastic.xlsx", engine='openpyxl')
head_df = df.head(5).to_string(index=False)

description_df=""" 
    Este DataFrame contiene información de obserbabilidad de servidores

    Las columnas incluyen: 
    - *Name*: Nombre del host.
    - *IP*: Dirección IP del host.
    - *Date*: Fecha de la medición.
    - *Memory Min*: valor minimo de consumo de memoria RAM 
    - *Memory Max*: valor máximo de consumo de memoria RAM 
    - *Memory Avg*: valor promedio de consumo de memoria RAM 
    - *Disk Min*: valor minimo de consumo de disco duro 
    - *Disk Max*: valor máximo de consumo de disco duro 
    - *Disk Avg*: valor promedio de consumo de disco duro 
    - *CPU Min*: valor minimo de consumo de CPU 
    - *CPU Max*: valor máximo de consumo de CPU 
    - *CPU Avg*: valor promedio de consumo de CPU 
    - *GPU Min*: valor minimo de consumo de GPU 
    - *GPU Max*: valor máximo de consumo de GPU 
    - *GPU Avg*: valor promedio de consumo de GPU 

    A continuación, se muestran las primeras 5 filas del DataFrame:

{head_df}
"""

# ==============================================================================================
# Function
# ==============================================================================================

def handle_query(query: str) -> bool:
    keywords = ['gráfico', 'dibuja', 'muestra', 'plot', 'visualización', 'visualizar', 'visualizando', 'visualiza', 'diagramar', 'diagrama', 'diagramando', 'dibujar', 'dibujo', 'dibujando',  'ploteo', 'plotea', 'ploteando', 'esquematizar', 
                'esquema', 'esquematizando', 'ilustrar', 'ilustra', 'ilustración', 'ilustrando', 'mostrar', 'muestra', 'mostrando', 'graficar', 'grafica', 'graficando', 'trazar', 'traza', 'trazando', 'bosquejar', 'bosquejo', 'bosquejando']
    
    if any(keyword in query.lower() for keyword in keywords):
        return True
    else:
        return False

# ----------------------------------------------------------------------------------------------

def observability_plot(query: str, chat_id: str) -> (str, bool):

    data_folder = "/Users/guane/Documentos/Codes/ChatBot-RAG/plots"

    # Asegura que el directorio existe
    os.makedirs(f"{data_folder}/{chat_id}", exist_ok=True)  
    number_image = len(os.listdir(f"{data_folder}/{chat_id}"))
    path_plot = f"{data_folder}/{chat_id}/output_{number_image}.jpg"


    query=f"""{query}. Construye el gráfico de la forma más estética posible para mostrar a un usuario. 
    Puedes utilizar los siguientes colores: verde y gris en diferentes tonalidades (si es necesario, utiliza más colores).
    Además, los títulos y ejes de los gráficos deben estar en español.
    Guarda la imagen en la ruta relativa {path_plot}.
    No ejecutes el comando plt.show().
    Siempre ejecuta el comando plt.tight_layout().
    """

    agent = create_pandas_dataframe_agent(
        llm=LLM, 
        df=df, 
        verbose=True,
        agent_type="openai-functions",
        prefix=description_df,
        allow_dangerous_code=True,
        include_df_in_prompt=True, 
        language="es")

    try:
        response = agent.invoke(query).get("output", "")
        flag_image = True
    except Exception as e:
        response = "No fue posible procesar la solicitud. Detalle del error: " + str(e)
        flag_image = False
    
    return response, flag_image

# ----------------------------------------------------------------------------------------------

def observability_question(query: str, chat_id: str) -> (str, bool):

    ejemplos = [
        {"pregunta": "¿Qué información posees?",
        "respuesta": "Poseo información sobre recursos de computo como el Nombre, la dirección IP, Fecha, y métricas de consumo de CPU, memoria, Disco y GPU"},
        {"pregunta": "¿Cuál es el valor máximo de consumo de CPU registrado?",
        "respuesta": "El valor máximo de consumo de CPU registrado es de X GHz."},
        {"pregunta": "Dame el promedio del consumo de memoria RAM durante el último mes.",
        "respuesta": "El promedio de consumo de memoria RAM durante el último mes es de X GB."}
    ]

    TEMPLATE_INIT = """Eres un asistente llamado Clico especializado en el análisis de datos de obserbabilidad. Tu misión es ayudar a los compañeros de trabajo a mejorar su experiencia.
    
    Dejo algunos ejemplos de preguntas que he respondido antes y que debes tomar como referencia:
    Ejemplo 1:
    Pregunta: {{ejemplos[0]['pregunta']}}
    Respuesta: {{ejemplos[0]['respuesta']}}

    Ejemplo 2:
    Pregunta: {{ejemplos[1]['pregunta']}}
    Respuesta: {{ejemplos[1]['respuesta']}}

    Ejemplo 3:
    Pregunta: {{ejemplos[2]['pregunta']}}
    Respuesta: {{ejemplos[2]['respuesta']}}

    Si una pregunta no se relaciona directamente con estos datos, debes responder: 'Solo puedo proporcionar información relacionada con los recursos computacionales listados anteriormente.'
    """

    TEMPLATE = TEMPLATE_INIT.format(ejemplos=ejemplos)

    prompt_template = ChatPromptTemplate.from_messages(
       [("system", TEMPLATE),
         MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")]
    )

    agent = create_pandas_dataframe_agent(
        llm=LLM,                         # Model
        df=df,                           # DataFrame
        prefix=description_df,           # Añade la descripción al inicio del prompt
        prompt_template=prompt_template, 
        agent_type="openai-tools", 
        verbose=True, 
        include_df_in_prompt=True,
        umber_of_head_rows=5, 
        allow_dangerous_code=True,
        language="es")
    
    try:
        response = agent.invoke(query).get("output", "")
    except Exception as e:
        response = "No fue posible procesar la solicitud. Detalle del error: " + str(e)
    
    return response

# ==============================================================================================
# Interaction Loop
# ==============================================================================================
chat_id = "12345"
try:
    while True:
        user_input = input("Tú: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Saliendo del chat...")
            break
        
        value = handle_query(user_input)
        print(" =================== \n", value, "\n =================== ")
        if value == False:
            response = observability_question(user_input, chat_id)
        else:
            response, flag_image = observability_plot(user_input, chat_id)

        print("Bot:", response)




except KeyboardInterrupt:
    print("Chat finalizado abruptamente.")
