from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
## Langmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

## Prompt Template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

## streamlit framework

st.title('Langchain Demo With OPENAI API')
input_text=st.text_input("Search the topic u want")

# openAI LLm 
llm=ChatOpenAI(model="gpt-3.5-turbo")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# import streamlit as st
# import os
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# # Check if the environment variables are loaded correctly
# openai_api_key = os.getenv("OPENAI_API_KEY")
# langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

# # Print the environment variables to verify
# print(f"OPENAI_API_KEY: {openai_api_key}")
# print(f"LANGCHAIN_API_KEY: {langchain_api_key}")

# if openai_api_key is None:
#     raise ValueError("OPENAI_API_KEY environment variable not set")
# if langchain_api_key is None:
#     raise ValueError("LANGCHAIN_API_KEY environment variable not set")

# os.environ["OPENAI_API_KEY"] = openai_api_key
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

# ## Prompt Template
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant. Please respond to the user queries"),
#         ("user", "Question:{question}")
#     ]
# )

# ## Streamlit framework
# st.title('Langchain Demo With OPENAI API')
# input_text = st.text_input("Search the topic you want")

# # OpenAI LLM
# llm = ChatOpenAI(model="gpt-3.5-turbo")
# output_parser = StrOutputParser()
# chain = prompt | llm | output_parser

# if input_text:
#     st.write(chain.invoke({'question':input_text}))
