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
from langchain_core.output_parsers import StrOutputParser
from neo4j import GraphDatabase
import time

load_dotenv()

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

query = "Identify top three python engineers"

llm = ChatOpenAI(model="gpt-4o", temperature=0)

main_focus = "python"

cql_query = """
MATCH (n) 
WHERE n.name =~ "(?i).*python.*" 
RETURN n;
"""
all_python_related_nodes = []
with driver.session() as session:
    result = session.run(cql_query)
    for record in result:
        all_python_related_nodes.append(record)

print(all_python_related_nodes)

all_root_nodes = []
for node in all_python_related_nodes:

    node_name = node['n']._properties['name']
    cql_query_roots = f"""
    MATCH (n) 
    WHERE n.name = "{node_name}"
    MATCH (n)<-[*0..]-(root)
    WHERE NOT ( ()-->(root) )
    RETURN root
    """

    with driver.session() as session:
        result = session.run(cql_query_roots)
        for idx, record in enumerate(result):
            if idx == 0:
                all_root_nodes.append(record[0]._properties['name'])
            else:
                break

print("\nAll root nodes for the Python-related nodes:")
for node in all_root_nodes:
    print(node)
    
root_node = input("Enter the root node: ")

selected_all_python_related_nodes = []
for idx, node in enumerate(all_python_related_nodes):
    if all_root_nodes[idx] == root_node:
        selected_all_python_related_nodes.append(node)

print(selected_all_python_related_nodes)

