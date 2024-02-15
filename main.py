import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_together import Together


os.environ["TOGETHER_API_KEY"] = st.secrets["TOGETHER_API_KEY"]

tweet_template = """
Give me {number} tweets on {topic}.-
"""

tweet_prompt = PromptTemplate(template = tweet_template, input_variables = ['number', 'topic'])

# gpt3_model = ChatOpenAI(model_name = "gpt-3.5-turbo-0125")  # use "gpt-4-0125-preview" for GPT-4 model

llama_model = Together(
    model="meta-llama/Llama-2-70b-chat-hf",
    # model = "mistralai/Mistral-7B-Instruct-v0.2",
    # model = "meta-llama/Llama-2-70b-hf",
    temperature=0.7,
)

tweet_generator = LLMChain(prompt = tweet_prompt, llm = llama_model)

st.title("Tweet Generator 🐦")
st.subheader("🚀 Generate tweets on any topic")

topic = st.text_input("Topic")

number = st.number_input("Number of tweets", min_value = 1, max_value = 10, value = 1, step = 1)

if st.button("Generate"):
    tweets = tweet_generator.run(number = number, topic = topic)
    st.write(tweets)
    # for tweet in tweets:
    #     st.write(tweet)
