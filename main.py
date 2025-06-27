import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# LangSmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With Ollama"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

# Response Generation Function
def generate_response(question, llm_model, temperature, max_tokens):
    llm = Ollama(model=llm_model, temperature=temperature)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

# Title
st.title("🤖Chatbot💬")


# Sidebar Settings
llm_model = st.sidebar.selectbox("Select Open Source model", ["mistral"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)


# Input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

# Output
if user_input:
    try:
        response = generate_response(user_input, llm_model, temperature, max_tokens)
        st.markdown(f"**Bot:** {response}")
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
else:
    st.write("Please provide a question.")
