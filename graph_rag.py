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
    name1 = node1["name"] if "name" in node1 else next(iter(node1.items()))[1] if node1 else "unknown"
    name2 = node2["name"] if "name" in node2 else next(iter(node2.items()))[1] if node2 else "unknown"
    
    # Relationship type
    rel_type = relationship.type
    
    # Build the string
    return f'(:{label1} {{name: "{name1}"}})-[:{rel_type}]->(:{label2} {{name: "{name2}"}})'

def get_all_nodes_and_relationships(file_names_list):
    with driver.session() as session:
        nodes_list = []
        relationships_list = []
        node_properties = {}
        
        for file_name in file_names_list:
            # Updated query to get more complete subgraph with increased depth
            query = f"""MATCH (target:FILE {{ name: "{file_name}" }})
                    CALL apoc.path.subgraphAll(target, {{ maxLevel: 3 }}) YIELD nodes, relationships
                    RETURN nodes, relationships"""
            print(f"Executing query for file: {file_name}")
            result = session.run(query)
            
            record_count = 0
            for record in result:
                record_count += 1
                try:
                    for node in record[0]:
                        try:
                            # Extract node name and properties
                            if "name" in node._properties:
                                node_name = node._properties["name"]
                                nodes_list.append(node_name)
                                
                                # Store node properties for context enrichment
                                node_properties[node_name] = {
                                    "labels": list(node.labels),
                                    "properties": node._properties
                                }
                            else:
                                # If not, add the node with its available properties
                                print(f"Node without 'name' property found: {node._properties}")
                                prop_str = str(next(iter(node._properties.values()))) if node._properties else "unnamed_node"
                                nodes_list.append(prop_str)
                                
                                # Store properties even for nodes without name
                                node_properties[prop_str] = {
                                    "labels": list(node.labels),
                                    "properties": node._properties
                                }
                        except Exception as e:
                            print(f"Error accessing node properties: {e}")
                    
                    for rel in record[1]:
                        try:
                            rel_str = relationship_to_string(rel)
                            relationships_list.append(rel_str)
                        except Exception as e:
                            print(f"Error converting relationship to string: {e}")
                except Exception as e:
                    print(f"Error processing record: {e}")
            
            if record_count == 0:
                print(f"No records found for file: {file_name}")
                
                # Try alternative query without APOC to see if the file node exists
                check_query = f"""MATCH (target:FILE {{ name: "{file_name}" }})
                                RETURN target"""
                check_result = session.run(check_query)
                if not check_result.peek():
                    print(f"File node with name '{file_name}' does not exist in database!")
        
        # Remove duplicates while preserving order
        nodes_list = list(dict.fromkeys(nodes_list))
        relationships_list = list(dict.fromkeys(relationships_list))
        
        print(f"Retrieved {len(nodes_list)} nodes and {len(relationships_list)} relationships")
        return nodes_list, relationships_list, node_properties

def get_relationships_for_node(node_name):
    query = """
        MATCH (n {name: $node_name})-[r]-(m)
        RETURN n, r, m
    """
    try:
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
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        print("Checking Neo4j connection...")
        try:
            # Test basic connection
            with driver.session() as test_session:
                test_result = test_session.run("RETURN 1 AS test")
                test_result.single()
                print("Neo4j connection is working, but the specific query failed.")
        except Exception as conn_err:
            print(f"Neo4j connection test failed: {conn_err}")
            print("Please check that your Neo4j server is running and credentials are correct.")
        
        # Return empty data as fallback
        return [], []
    
def extract_main_node_chain(query, nodes, node_properties=None):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Enhanced prompt with node properties when available
    if node_properties:
        template = """You are a highly skilled assistant that specializes in extracting the main entity from a query.
        You are given a query and a list of all nodes in the graph database with their properties.
        Extract the SINGLE most relevant entity/node from the query that will be the focal point for answering it.
        
        Query: {query}
        List of nodes with properties: {node_info}
        
        Return ONLY the exact name of the main node as it appears in the list (no explanations):
        """
        node_info = "\n".join([f"{node}: {node_properties.get(node, {})}" for node in nodes[:30]])  # Limit to prevent token overflow
    else:
        template = """You are a highly skilled assistant that specializes in extracting the main entity from a query.
        You are given a query and a list of all nodes in the graph database.
        Extract the SINGLE most relevant entity/node from the query that will be the focal point for answering it.
        
        Query: {query}
        List of nodes: {nodes}
        
        Return ONLY the exact name of the main node as it appears in the list (no explanations):
        """
        node_info = nodes
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["query", "node_info"] if node_properties else ["query", "nodes"]
    )
    
    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({
        "query": query, 
        "node_info" if node_properties else "nodes": node_info
    })
    
    # Clean up the result to ensure it matches a node in the list
    result = result.strip()
    if result not in nodes:
        # Try finding closest match if exact match not found
        for node in nodes:
            if result.lower() in node.lower():
                return node
    
    return result
    
def rephrase_query_chain(query, nodes, relationships, node_properties=None):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    
    # Enhanced prompt that gives better guidance on query reformulation
    template = """You are a knowledge graph query specialist that reformulates natural language questions into precise queries that can be executed against a graph database.

    Given the user's original question, transform it into a more specific query that:
    1. References the exact entity names as they appear in the knowledge graph
    2. Specifies the types of relationships to look for
    3. Includes any relevant constraints or conditions
    4. Is formulated to be answered by traversing the graph structure

    Available nodes in graph: {nodes}
    Available relationships in graph: {relationships}
    
    Original question: {query}

    Reformulated query (make it specific but natural language):
    """
    
    chain = prompt = PromptTemplate(
        template=template,
        input_variables=["nodes", "relationships", "query"]
    ) | llm
    
    return chain.invoke({"nodes": nodes[:50], "relationships": relationships[:50], "query": query})

def generate_optimized_cypher(query, schema, file_names_list):
    """Generate an optimized Cypher query based on the user query and schema"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    template = """
    You are a Neo4j Cypher expert. Generate the most efficient Cypher query to answer this question.
    
    Question: {query}
    
    Database schema:
    {schema}
    
    The query should:
    1. Be limited to these specific files: {file_names_list}
    2. Use appropriate MATCH patterns and WHERE clauses
    3. Include RETURN statements that directly answer the question
    4. Limit results to relevant information only
    5. Use efficient traversal patterns (avoiding cartesian products)
    
    Example query structure:
    ```
    MATCH (file:FILE {{name: $fileName}})
    MATCH path = (file)-[*1..3]-(relevantNode)
    WHERE relevantNode.property = 'value'
    RETURN relevantNode.name, relevantNode.property
    ```
    
    Write ONLY the Cypher query without explanations:
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["query", "schema", "file_names_list"]
    )
    
    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({
        "query": query,
        "schema": schema,
        "file_names_list": file_names_list
    })
    
    return result

def enrich_results_with_context(results, node_properties):
    """Enrich query results with node context information"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    template = """
    You are an information synthesis expert. Enhance the following database query result with relevant context about the entities mentioned.
    
    Original result:
    {result}
    
    Additional context about entities:
    {context}
    
    Provide a comprehensive answer that integrates the query result with the additional context.
    Keep your response concise and focused on answering the original question with the enriched information:
    """
    
    # Extract entity names from the results
    import re
    entity_pattern = r'"([^"]+)"'
    entities = re.findall(entity_pattern, results)
    
    # Build context from node properties
    context = {}
    for entity in entities:
        if entity in node_properties:
            context[entity] = node_properties[entity]
    
    if not context:
        return results
        
    prompt = PromptTemplate(
        template=template,
        input_variables=["result", "context"]
    )
    
    chain = prompt | llm
    
    enriched_result = chain.invoke({
        "result": results,
        "context": context
    })
    
    return enriched_result.content

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

schema = graph.get_schema

template = """
Task: Generate a Cypher statement to query the graph database.

You will be given a rephrased query and a list of file names.
Generate a cypher statement to answer the rephrased query.

Use the file name list to filter the nodes and relationships.
file_names_list: {file_names_list}

Use this query as an example to generate the Cypher statement.
```
MATCH (target:FILE {{name: $fileName}})
CALL apoc.path.subgraphAll(target, {{maxLevel: 3}}) YIELD nodes, relationships
WITH nodes, relationships
UNWIND nodes as node
WHERE node.property = 'value'
RETURN node.name, node.property
```

Instructions:
Use only relationship types and properties provided in schema.
Do not use other relationship types or properties that are not provided.
The query should be optimized for performance and relevance.

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

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

qa = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=question_prompt,
    verbose=True,
    allow_dangerous_requests=True,
    return_intermediate_steps=True
)

file_names_list = [
    'Bob Smith 1.pdf',
    'Evan Patel 1.pdf',
    'Fatima_Kiyani.pdf',
    'Fiona Zhang 1.pdf',
    'George Kim 1.pdf',
    'Hasnain Ali Resume.pdf',
    'Immar_Karim 1.pdf',
    'Muhammad Faris Khan CV.pdf',
    'Raza Ali Poonja - Resume.pdf'
]
list_of_all_nodes, list_of_all_relationships, node_properties = get_all_nodes_and_relationships(file_names_list)

while True:
    question = input("Enter a question: ")
    
    if question in ["/q", "/quit", "/exit", "/stop", "/end", "/close", "/bye", "/goodbye", "/byebye", "/goodbyebye", "/goodbyecya"]:
        break
    
    # Extract main entity from the query
    main_node = extract_main_node_chain(question, list_of_all_nodes, node_properties)
    print(f"Main entity identified: {main_node}")
    
    # Get detailed relationship information
    records, relationships = get_relationships_for_node(main_node)
    
    # Rephrase the query using knowledge of the main node and its relationships
    rephrased_query = rephrase_query_chain(
        question, 
        list_of_all_nodes, 
        list_of_all_relationships, 
        node_properties
    )
    print(f"Rephrased query: {rephrased_query.content}")
    
    # Try multiple approaches if needed
    retries = 3
    for attempt in range(retries):
        try:
            # First approach: Use the GraphCypherQAChain
            result = qa.invoke({
                "query": rephrased_query.content, 
                "file_names_list": file_names_list
            })
            
            # Second approach if first one fails: Generate optimized Cypher directly
            if not result["result"] or len(result["result"]) < 10:
                custom_cypher = generate_optimized_cypher(
                    rephrased_query.content,
                    schema,
                    file_names_list
                )
                
                # Execute the custom cypher
                with driver.session() as session:
                    cypher_result = session.run(custom_cypher)
                    result_data = [dict(record) for record in cypher_result]
                    
                    if result_data:
                        result["result"] = str(result_data)
            
            # Enrich the results with additional context
            enriched_result = enrich_results_with_context(result["result"], node_properties)
            print(enriched_result)
            break
            
        except Exception as e:
            if attempt == retries - 1:
                print(f"Failed after {retries} attempts. Error: {e}")
            else:
                print(f"Attempt {attempt + 1} failed. Retrying...")
                time.sleep(1)  # Short delay before retry