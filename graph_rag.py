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

load_dotenv()

def get_all_nodes():
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN n")
        nodes = [record["n"]._properties["name"] for record in result]
        return nodes

def get_relationships_for_node(node_name):
    query = """
        MATCH (n {name: $node_name})-[r]-(m)
        RETURN n, r, m
    """
    with driver.session() as session:
        result = session.run(query, node_name=node_name)
        
        records = []
        relationships = []
        for record in result:
            records.append({
                "node": record["n"],
                "relationship": record["r"],
                "connected_node": record["m"]
            })
            relationships.append(record[1].type)
        return records, relationships
    
def extract_main_node_chain(query, nodes):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = PromptTemplate(
        template=""" You are a highly skilled assistant that specializes in extracting the main node from a query.
        You are given a query and a list of all nodes in the graph database.
        You need to extract the main node from the query.
        The main node is the node that is most relevant to the query.
        
        Note: Only return the name of the node.
        
        Query: {query}
        List of all nodes: {nodes}
        
        Main node:
        """
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query, "nodes": nodes})
    
def rephrase_query_chain(query, main_node, relationships):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = PromptTemplate(
        template=""" You are a highly skilled assistant that specializes in rephrasing user queries using the main node and the relationships of the main node in the graph database. 

        The rephrased query should include the main node and the relevant relationship of the main node.

        List of relationships: {relationships}
        Query: {query}

        Rephrased Query:
        """
    )
    
    chain = prompt | llm
    return chain.invoke({"relationships": relationships, "query": query})

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
    doc_class = input("Enter the class of the document: ['RESUME', 'SCIENCE_ARTICLE', 'TECHNICAL_DOCUMENT']")
    
    if question in ["/q", "/quit", "/exit", "/stop", "/end", "/close", "/bye", "/goodbye", "/byebye", "/goodbyebye", "/goodbyecya"]:
        break
    
    main_node = extract_main_node_chain(question, list_of_all_nodes)
    records, relationships = get_relationships_for_node(main_node)
    
    rephrased_query = rephrase_query_chain(question, main_node, relationships)
    result = qa.invoke({"query": rephrased_query.content})
    print(result["result"])
