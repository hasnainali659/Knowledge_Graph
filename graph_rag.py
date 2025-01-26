import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from neo4j import GraphDatabase

load_dotenv()

def get_all_nodes():
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN n")
        nodes = [record["n"]._properties["name"] for record in result]
        return nodes
    
def rephrase_query_chain(query, nodes):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = PromptTemplate(
        template=""" You are a highly skilled assistant that specializes in rephrasing user queries to match the names (exact case) of relevant nodes in a graph database. 

        Your task:
        1. You are given a list of all possible nodes in the database.
        2. You are given the user's query.
        3. You must rephrase the query so that any references to a node are replaced with the exact node name (maintaining the correct case) from the provided list.
        4. Ensure the essential meaning and intent of the user’s query remains the same.
        5. If no node in the list is relevant, leave the query as-is.
        6. When multiple nodes seem relevant, select the single best match that most closely aligns with the user’s context or question.

        Key points:
        - Maintain correct casing and spelling of nodes exactly as they appear in the list.
        - Only replace references that clearly map to a node in the list.
        - Do not fabricate nodes or alter other parts of the query unnecessarily.
        - Make sure the final rephrased query is concise, clear, and grammatically correct.

        Below are some examples:

        Example 1:
        - Query: "Who has certification in python programming?"
        - List of all nodes: ['Python', 'Machine Learning', 'Data Analysis', 'Programming in Python']
        - Rephrased Query: "Who has certification in Programming in Python?"

        Example 2:
        - Query: "Is there any course on machine learning?"
        - List of all nodes: ['Machine Learning', 'Deep Learning', 'Introduction to AI']
        - Rephrased Query: "Is there any course on Machine Learning?"

        Example 3:
        - Query: "Do we have any advanced data analysis projects?"
        - List of all nodes: ['Data Analysis', 'Data Engineering', 'Advanced Data Analysis Methods']
        - Rephrased Query: "Do we have any advanced Data Analysis projects?"

        Now, apply these instructions to the real data:

        List of all nodes: {nodes}
        Query: {query}

        Rephrased Query:
        """
    )
    
    chain = prompt | llm
    return chain.invoke({"nodes": nodes, "query": query})

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI_1"),
    username=os.getenv("NEO4J_USERNAME_1"),
    password=os.getenv("NEO4J_PASSWORD_1")
)

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI_1"),
    auth=(os.getenv("NEO4J_USERNAME_1"), os.getenv("NEO4J_PASSWORD_1"))
)

schema = graph.get_schema

template = """
Task: Generate a Cypher statement to query the graph database.

Instructions:
Use only relationship types and properties provided in schema.
Do not use other relationship types or properties that are not provided.

schema:
{schema}

Note: Do not include explanations or apologies in your answers.
Do not answer questions that ask anything other than creating Cypher statements.
Do not include any text other than generated Cypher statements.

Question: {question}""" 

question_prompt = PromptTemplate(
    template=template, 
    input_variables=["schema", "question"] 
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

qa = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=question_prompt,
    verbose=True,
    allow_dangerous_requests=True
)
list_of_all_nodes = get_all_nodes()

while True:
    question = input("Enter a question: ")
    if question in ["/q", "/quit", "/exit", "/stop", "/end", "/close", "/bye", "/goodbye", "/byebye", "/goodbyebye", "/goodbyecya"]:
        break
    rephrased_query = rephrase_query_chain(question, list_of_all_nodes)
    result = qa.invoke({"query": rephrased_query.content})
    print(result["result"])
