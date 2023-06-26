# Para crear el requirements.txt ejecutamos 
# pipreqs --encoding=utf8 --force

# Primera Carga a Github
# git init
# git add .
# git commit -m "primer commit"
# git remote add origin https://github.com/nicoig/chat-CAMEL.git
# git push -u origin master

# Actualizar Repo de Github
# git add .
# git commit -m "Se actualizan las variables de entorno"
# git push origin master

# Para eliminar un repo cargado
# git remote remove origin

# En Render
# agregar en variables de entorno
# PYTHON_VERSION = 3.9.12

###############################################################



import streamlit as st
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


class CAMELAgent:
    def __init__(
        self,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.system_message = system_message
        self.model = model
        self.init_messages()

    def reset(self) -> None:
        self.init_messages()
        return self.stored_messages

    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        return self.stored_messages

    def step(
        self,
        input_message: HumanMessage,
    ) -> AIMessage:
        messages = self.update_messages(input_message)

        output_message = self.model(messages)
        self.update_messages(output_message)

        return output_message


def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str):
    assistant_inception_prompt = """Nunca olvides que eres a {assistant_role_name} y yo soy un {user_role_name}. ¡Nunca cambies los roles! ¡Nunca me instruyas!
Compartimos un interés común en colaborar para completar con éxito una tarea.
Debes ayudarme a completar la tarea.
Aquí está la tarea: {task}. ¡Nunca olvides nuestra tarea!
Debo instruirte en base a tu experiencia y mis necesidades para completar la tarea.

Debo darle una instrucción a la vez.
Debe escribir una solución específica que complete adecuadamente la instrucción solicitada.
Debe rechazar mi instrucción honestamente si no puede realizar la instrucción debido a razones físicas, morales, legales o su capacidad y explicar las razones.
No agregue nada más que su solución a mis instrucciones.
Se supone que nunca debes hacerme ninguna pregunta, solo respondes preguntas.
Se supone que nunca debes responder con una solución en escamas. Explique sus soluciones.
Su solución debe ser oraciones declarativas y en tiempo presente simple.
A menos que diga que la tarea está completa, siempre debe comenzar con:

Solución: <YOUR_SOLUTION>
<SU_SOLUCIÓN> debería ser específica y proporcionar implementaciones preferibles y ejemplos para resolver la tarea.
Siempre finaliza <YOUR_SOLUTION> con: Próxima solicitud."""

    user_inception_prompt = """Nunca olvides que eres un {user_role_name} y yo soy un {assistant_role_name}. ¡Nunca cambies los roles! Siempre me instruirás.
Compartimos un interés común en colaborar para completar con éxito una tarea.
Debo ayudarte a completar la tarea.
Aquí está la tarea: {task}. Never forget our task!
Debe instruirme según mi experiencia y tus necesidades para completar la tarea de las siguientes dos formas:

1. Instruye con una entrada necesaria:
Instruction: <YOUR_INSTRUCTION>
Input: <YOUR_INPUT>

2. Instruir sin ninguna entrada:
Instruction: <YOUR_INSTRUCTION>
Input: None

La "Instruction" describe una tarea o pregunta. La "Entrada" emparejada proporciona más contexto o información para la solicitud "Instruction".

Debes darme una instrucción a la vez.
Debo escribir una respuesta que complete adecuadamente la instrucción solicitada.
Debo rechazar tu instrucción honestamente si no puedo realizar la instrucción debido a razones físicas, morales, legales o mi capacidad, y explicar las razones.
Debes instruirme, no hacerme preguntas.
Comienza a instruirme ahora utilizando las dos formas descritas anteriormente.
¡No agregues nada más que tu instrucción y la entrada correspondiente opcional!
Sigue dándome instrucciones y entradas necesarias hasta que consideres que la tarea está completa.
Cuando la tarea esté completa, solo debes responder con una sola palabra <CAMEL_TASK_DONE>.
Nunca digas <CAMEL_TASK_DONE> a menos que mis respuestas hayan resuelto tu tarea."""

    assistant_sys_template = SystemMessagePromptTemplate.from_template(
        template=assistant_inception_prompt
    )
    assistant_sys_msg = assistant_sys_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task,
    )[0]

    user_sys_template = SystemMessagePromptTemplate.from_template(
        template=user_inception_prompt
    )
    user_sys_msg = user_sys_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task,
    )[0]

    return assistant_sys_msg, user_sys_msg


def main():
    st.title("Aplicación de Chat CAMEL")

    assistant_role_name = st.text_input("Rol del asistente:", value="Desarrollador experto en Python")
    user_role_name = st.text_input("Rol del usuario:", value="Gerente de Innovación e Investigación")
    task = st.text_area("Tarea a resolver:", value="Desarrollar una aplicación solucionando problemas cotidianos de las empresas de minería con alto impacto utilizando el modelo de lenguaje Python y los modelos de langchain, utilizando los agentes especializados en temas.")
    num_iterations = st.number_input("Cantidad de iteraciones:", value=10, min_value=1)

    if st.button("Iniciar Chat"):
        if assistant_role_name and user_role_name and task:
            assistant_sys_msg, user_sys_msg = get_sys_msgs(assistant_role_name, user_role_name, task)
            assistant_agent = CAMELAgent(assistant_sys_msg, ChatOpenAI(temperature=0.2))
            user_agent = CAMELAgent(user_sys_msg, ChatOpenAI(temperature=0.2))

            # Reset agents
            assistant_agent.reset()
            user_agent.reset()

            # Initialize chats
            assistant_msg = HumanMessage(
                content=(
                    f"{user_sys_msg.content}. "
                    "Ahora empieza a darme presentaciones una por una.. "
                    "Solo responde con Instruction y el Input."
                )
            )

            user_msg = HumanMessage(content=f"{assistant_sys_msg.content}")
            user_msg = assistant_agent.step(user_msg)

            chat_turn_limit, n = 30, 0
            chat_log = []  # Variable para almacenar la conversación
            while n < chat_turn_limit and n < num_iterations:
                n += 1
                user_ai_msg = user_agent.step(assistant_msg)
                user_msg = HumanMessage(content=user_ai_msg.content)
                st.text(f"Usuario de IA ({user_role_name}):\n\n{user_msg.content}\n\n")

                assistant_ai_msg = assistant_agent.step(user_msg)
                assistant_msg = HumanMessage(content=assistant_ai_msg.content)
                st.text(f"Asistente de IA ({assistant_role_name}):\n\n{assistant_msg.content}\n\n")
                if "<CAMEL_TASK_DONE>" in user_msg.content:
                    break

                # Guardar los mensajes en el registro de la conversación
                chat_log.append(f"Usuario de IA ({user_role_name}):\n\n{user_msg.content}\n\n")
                chat_log.append(f"Asistente de IA ({assistant_role_name}):\n\n{assistant_msg.content}\n\n")

            # Botón para descargar la conversación
            if len(chat_log) > 0:
                st.markdown("---")
                st.subheader("Descargar Conversación")
                download_text = "\n".join(chat_log)
                st.download_button(
                    label="Descargar Conversación",
                    data=download_text,
                    file_name="conversacion.txt",
                    mime="text/plain",
                )


if __name__ == "__main__":
    main()
