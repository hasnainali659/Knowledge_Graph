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

def relationship_to_string(relationship):
    """Convert a Neo4j Relationship into a string like:
       (:Entity {name: "Hasnain Ali Poonja"})-[:HAS_EDUCATION]->(:Entity {name: "NUST SMME"})
    """
    # Extract the start and end nodes
    node1, node2 = relationship.nodes
    
    # (Optional) Convert labels to a single label string if multiple labels exist
    label1 = list(node1.labels)[0] if node1.labels else ""
    label2 = list(node2.labels)[0] if node2.labels else ""
    
    # Retrieve the node 'name' property
    name1 = node1["name"]
    name2 = node2["name"]
    
    # Relationship type
    rel_type = relationship.type
    
    # Build the string
    return f'(:{label1} {{name: "{name1}"}})-[:{rel_type}]->(:{label2} {{name: "{name2}"}})'

def get_all_nodes_and_relationships(file_names_list):
    with driver.session() as session:
        nodes_list = []
        relationships_list = []
        for file_name in file_names_list:
            query = f"""MATCH (target:File {{ name: "{file_name}" }})
                    CALL apoc.path.subgraphAll(target, {{ maxLevel: 2 }}) YIELD nodes, relationships
                    RETURN nodes, relationships"""
            result = session.run(query)
            for record in result:
                nodes = [node._properties["name"] for node in record[0]]
                nodes_list.extend(nodes)
                relationship_string = [relationship_to_string(relationships) for relationships in record[1]]
                relationships_list.extend(relationship_string)

        return nodes_list, relationships_list

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
    
def rephrase_query_chain(query, nodes, relationships):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = PromptTemplate(
        template=""" You are a highly skilled assistant that specializes in rephrasing user queries using the list of nodes and the 
        list of relationships in the graph database. 

        The rephrased query should include the exact node and the relevant relationship of the node from the list of nodes and the list of relationships.

        List of nodes: {nodes}
        List of relationships: {relationships}
        Query: {query}

        Rephrased Query:
        """
    )
    
    chain = prompt | llm
    return chain.invoke({"nodes": nodes, "relationships": relationships, "query": query})

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

You will be given a rephrased query and a list of file names.
Generate a cypher statement to answer the rephrased query.

Use the file name list to filter the nodes and relationships.
file_names_list: {file_names_list}

Use this query as an example to generate the Cypher statement.
"MATCH (target:File {{name: 'Evan Patel 1.pdf'}})
CALL apoc.path.subgraphAll(target, {{maxLevel: 6}}) YIELD nodes, relationships
RETURN nodes, relationships"

Instructions:
Use only relationship types and properties provided in schema.
Do not use other relationship types or properties that are not provided.

schema:
{schema}

Note: Do not include explanations or apologies in your answers.
Do not answer questions that ask anything other than creating Cypher statements.
Do not include any text other than generated Cypher statements.

Rephrased Query: {query}

Cypher Statement:
""" 

question_prompt = PromptTemplate(
    template=template, 
    input_variables=["schema", "query", "file_names_list"] 
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

qa = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=question_prompt,
    verbose=True,
    allow_dangerous_requests=True,
    return_intermediate_steps=True
)

file_names_list = ['Evan Patel 1.pdf', 
                   'George Kim 1.pdf',
                   'Evaluation-of-ECG-based-Recognition-of-Cardiac-Abnormalities-using-Machine-Learning-and-Deep-Learning.pdf',
                   'Motor_Parametric_Calculations_for_Robot.pdf']
list_of_all_nodes, list_of_all_relationships = get_all_nodes_and_relationships(file_names_list)

while True:
    question = input("Enter a question: ")
    # doc_class = input("Enter the class of the document: ['RESUME', 'SCIENCE_ARTICLE', 'TECHNICAL_DOCUMENT']")
    
    if question in ["/q", "/quit", "/exit", "/stop", "/end", "/close", "/bye", "/goodbye", "/byebye", "/goodbyebye", "/goodbyecya"]:
        break
    
    # main_node = extract_main_node_chain(question, list_of_all_nodes)
    # records, relationships = get_relationships_for_node(main_node)
    
    rephrased_query = rephrase_query_chain(question, list_of_all_nodes, list_of_all_relationships)
    retries = 3
    for attempt in range(retries):
        try:
            result = qa.invoke({"query": rephrased_query.content, "file_names_list": file_names_list})
            print(result["result"])
            break
        except Exception as e:
            if attempt == retries - 1:
                print(f"Failed after {retries} attempts. Error: {e}")
            else:
                print(f"Attempt {attempt + 1} failed. Retrying...")