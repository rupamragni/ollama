from langchain_ollama import ChatOllama
from langchain_core.prompts import (
                                        SystemMessagePromptTemplate,
                                        HumanMessagePromptTemplate,
                                        ChatPromptTemplate,
                                        PromptTemplate,
                                        MessagesPlaceholder
)

from langchain_core.output_parsers import StrOutputParser


base_url="http://localhost:11434"
model = 'llama3.2:3b'

llm = ChatOllama(
    base_url=base_url,
    model = model)

system = SystemMessagePromptTemplate.from_template("""You are helpful AI assistant who answer user questions based on the provided context. """)

prompt= """Answer user questions based on provided context ONLY ! If do not know the answer, just say  "I don't know".
            ### Context :
            {context}
             
            ### Question:
            {question}
             
            ### Answer :  """

human =HumanMessagePromptTemplate.from_template(prompt)

messages= [system,human]
template= ChatPromptTemplate(messages)

qna_chain= template | llm | StrOutputParser()

def ask_llm(context,question):
    return qna_chain.invoke({'context': context, 'question': question})