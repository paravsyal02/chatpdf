from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()

# Get the API key from the environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Optionally, set it in the environment
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# Initialize the LLM (make sure you configure it properly)
llm = ChatGoogleGenerativeAI(model="gemini-pro", api_key=GEMINI_API_KEY)

# Define the prompt template for generating questions
question_template = PromptTemplate(input_variables=["topic"], template="Generate a list of questions about {topic}.")

# Use LLMChain instead of RunnableSequence
question_chain = LLMChain(llm=llm, prompt=question_template)

print("Welcome to the Unlimited Question Generator!")
print("You can request questions on any topic once. Press 'Ctrl + C' to terminate the program.\n")

# Ask for the topic once
topic = input("Enter the topic for the questions: ")

# Variable to track question number
question_number = 1

# Run the loop to generate multiple questions for the same topic
while True:
    # Generate question
    response = question_chain.run({"topic": topic})
    
    print(f"\nGenerated Question {question_number}:")
    print(f"**Question {question_number}:**\n\n{response}\n")
    
    # Increment the question number
    question_number += 1