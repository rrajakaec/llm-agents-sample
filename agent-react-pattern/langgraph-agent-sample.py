import os

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# set up API key
os.environ["TAVILY_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

#Reference https://learn.deeplearning.ai/courses/ai-agents-in-langgraph/lesson/3/langgraph-components

tool = TavilySearchResults(max_results=2)
print(type(tool))
print(tool.name)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    
class Agent:
    
    def __init__(self, model, tools, system=""):
        self.system = system
        
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)
    
    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}
    
    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], 
                                       name=t['name'], 
                                       content=str(result)))
        print("Back to the model!")
        return {'messages': results}
    
    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0
    
prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!"""

model = ChatOpenAI(model="gpt-4-turbo") #reduce inference cost
abot = Agent(model, [tool], system=prompt)

#Visualize the graph we created
#from IPython.display import Image
#Image(abot.graph.get_graph().draw_png())

#Query 1
"""
messages = [HumanMessage(content="What is the weather in sf?")]
#esult = abot.graph.invoke({'messages': messages})
print(result)
print(result['messages'][-1].content)
"""

#Query 2
"""
messages = [HumanMessage(content="What is the weather in SF and NJ?")]
result = abot.graph.invoke({'messages': messages})
print(result)
print(result['messages'][-1].content)
"""

#Query 3
messages = [HumanMessage(content="Who won the super bowl in 2024? In what state is the winning team headquarters located? \
What is the GDP of that state? Answer each question.")]
result = abot.graph.invoke({'messages': messages})
#print(result)
print(result['messages'][-1].content)
