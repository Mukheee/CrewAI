import os
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun

# Set gemini pro llm
llm = ChatGoogleGenerativeAI(model='gemini-pro',
                             verbose=True,
                             temperature=0.6,
                             google_api_key="AIzaSyB-QKp-1YZSTw0Hdcvn5FMwIWSHV7wFjz4")

# Provide a search tool
search_tool = DuckDuckGoSearchRun()

# Define the Agents
# Define an Agent with roles and goals
# Agent1 (Researcher): Develop ideas for teaching someone
Researcher = Agent(
    role='Develop ideas for teaching someone new to Data science',
    goal='To make the student be able to answer basic to medium questions concerning Data Science',
    backstory="""You are a teacher that want to impact knowledge on students who are new to Data Science.
                   You are well versed in Data Science and the emerging trends concerning it.
                   You ou have a knack for analyzing and dissecting complex situations concerning data.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,  # using google gemini pro API
)

Writer = Agent(
    role='Use the researchers ideas to write a piece of text to explain the topic',
    goal='Craft or idealize a compelling text on Neural networks',
    backstory=""" You are a student of knowledge, known for being thirsty for knowledge on new things.
                    You then decided to dabble in Data science """,
    verbose=True,
    allow_delegation=True,
    llm=llm,  # using google gemini pro API
)

Examiner = Agent(
    role='Craft 2-3 test questions to evaluate understanding of the created text, along with the correct answer',
    goal='To test the student on his knowledge on Neural network',
    backstory="""You are a certified Data Scientist who examines people taking the professional course exam""",
    verbose=True,
    allow_delegation=True,
    llm=llm,
    tools=[]
)

# Tasks to perform
task1 = Task(
    description="Develop ideas for teaching some that is new to Data science",
    agent=Researcher
)

task2 = Task(
    description="Use the Researcher's ideas to write a piece of comprehensive text to explain a topic in Data science",
    agent=Writer
)

task3 = Task(
    description="Craft two or three questions to evaluate understanding of the created text, along with the corrected "
                "answers",
    agent=Examiner
)

# Create a crew

crew = Crew(
    agents=[Researcher, Writer, Examiner],
    tasks=[task1, task2, task3],
    verbose=2,
    process=Process.sequential
)

# get the crew to work

result = crew.kickoff()

print(result)
