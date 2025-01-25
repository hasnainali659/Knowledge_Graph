from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field

import PyPDF2
from main import Neo4jConnection

import os

path = 'docs/Bob Smith 1.pdf'

with open(path, 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text()

class ResumeContentSchema(BaseModel):
    entities: list[str] = Field(default=[], description="The list of entities in the resume")
    relationships: list[str] = Field(default=[], description="The list of relationships between entities in the resume")
    file_name: str = Field(default='', description="The name of the file")
    doc_class: str = Field(default='', description="The class of the document")
    cypher_queries: list[str] = Field(default=[], description="The list of cypher queries to create the knowledge graph")

    
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
parser = PydanticOutputParser(pydantic_object=ResumeContentSchema)

template = """
You are a resume parser. You are given a resume, the name of the file, the class of the document and you need to extract entities, relationships and cypher
queries to create a knowledge graph.

The class of the document is {doc_class}.
The name of the file is {file_name}.

First create a cypher query to create a root node named 'resume'. 
Then create a cypher query to create a node for the file name and link it to the resume node. via the relationship 'BELONGS_TO'.
Then create a cypher query to create a node for each entity and link it to the file node.

output format example:
{{
    'entities': ['Bob Smith', 'John Doe', 'Jane Doe', 'University of California, Los Angeles', 'Google', 'Software Engineer', 'Python', 'Java', 'SQL'],
    'relationships':['HAS_EDUCATION', 'HAS_EXPERIENCE', 'HAS_SKILLS', 'LIVES_IN', 'WORKS_AT'],
    'cypher_queries': ['MERGE (e:Entity {{name: $name}}) RETURN e', 'MERGE (r:Relationship {{name: $name}}) RETURN r']
}}

resume:
{text}
"""

prompt = PromptTemplate(template=template)
chain = prompt | llm
response = chain.invoke({"text": full_text, "doc_class": "resume", "file_name": "Bob Smith 1.pdf"})

response_content = response.content if hasattr(response, 'content') else response

try:
    parsed_response = parser.parse(response_content)
except OutputParserException:
    new_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
    parsed_response = new_parser.parse(response_content)
    
entities = parsed_response.entities
relationships = parsed_response.relationships
cypher_queries = parsed_response.cypher_queries

NEO4J_URI = os.getenv('NEO4J_URI_1')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME_1')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD_1')

neo4j_connection = Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

# create a root node name 'resume'
# resume_query = """MERGE (r:Resume {name: 'resume'}) RETURN r"""

# resume_query_parameters = {
#     'name': 'resume'
# }

# neo4j_connection.write_transaction(resume_query, resume_query_parameters)

# #create a node for file name and link it to the resume node
# file_query = """MERGE (f:File {name: $name}) RETURN f"""

# file_query_parameters = {
#     'name': 'Bob Smith 1.pdf'
# }

# neo4j_connection.write_transaction(file_query, file_query_parameters)

# # make a relationship between the file node and the resume node
# relationship_query = """MATCH (r:Resume), (f:File) WHERE r.name = 'resume' AND f.name = 'Bob Smith 1.pdf' CREATE (f)-[:BELONGS_TO]->(r) RETURN r, f"""

# neo4j_connection.write_transaction(relationship_query)

# # create a node for each entity and make a relationship between the entity node and the file node
# for entity in entities:
#     entity_query = """MERGE (e:Entity {name: $name}) RETURN e"""
#     entity_query_parameters = {
#         'name': entity
#     }
#     neo4j_connection.write_transaction(entity_query, entity_query_parameters)

# # make a relationship between the entity node and the file node
# for relationship in relationships:
#     relationship_query = """MATCH (e:Entity), (f:File) WHERE e.name = $entity AND f.name = 'Bob Smith 1.pdf' CREATE (e)-[:HAS_RELATIONSHIP]->(f) RETURN e, f"""
#     relationship_query_parameters = {
#         'entity': relationship
#     }
#     neo4j_connection.write_transaction(relationship_query, relationship_query_parameters)

for cypher_query in cypher_queries:
    neo4j_connection.write_transaction(cypher_query)



