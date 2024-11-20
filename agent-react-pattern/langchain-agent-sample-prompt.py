import os

#from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAI

#Reference https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent

# set up API key
os.environ["TAVILY_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

tools = [TavilySearchResults(max_results=1)]

# Get the prompt to use - you can modify this!
# prompt = hub.pull("hwchase17/react")
system_prompt = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer

Thought: you should always think about what to do

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer

Final Answer: the final answer to the original input question

Begin!

Question: {input}

Thought:{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(system_prompt)

# Choose the LLM to use
llm = OpenAI()

# Construct the ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "what is LangChain?"})