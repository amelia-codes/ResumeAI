#packages here
from langchain_community.utilities import SQLDatabase
from typing_extensions import TypedDict
import getpass
import os
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import Annotated
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph

if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
    os.environ["LANGSMITH_TRACING"] = "true"

#loading database
db_skills = SQLDatabase.from_uri("skills.sql")
db_abilities = SQLDatabase.from_uri("abilities.sql")
db_knowledge = SQLDatabase.from_uri("knowledge.sql")
db_work_activities = SQLDatabase.from_uri("activities.sql")
db_occupation_title = SQLDatabase.from_uri("occupation.sql")
db_words = SQLDatabase.from_uri("words.sql")
#db = the combined database
print(db.dialect)
print(db.get_usable_table_names())
#db.run("SELECT * FROM Artist LIMIT 10;")
#we want all words from occupation
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

user_prompt = "Enter x from y section: {input}"

query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)

for message in query_prompt_template.messages:
    message.pretty_print()

class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]

def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["input"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}
#add user occupation to prompt, query is list of allowed words
def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user input, corresponding SQL query, "
        "and SQL result, rewrite the user's input keeping approximately the" 
        "same number of characters (plus or minus x - same number of lines)."
        "Each word in the SQL query has an associated weight corresponding to" 
        "the word's score and a high-scoring response has a high score. While rewriting," 
        "you should use as many high weighted words in the query as"
        "possible while maintaining good sentence flow."
        "associated with the user's occupation \n\n"
        f"Question: {state['question']}\n"
        f"SQL Query: {state['query']}\n"
        f"SQL Result: {state['result']}"
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()
