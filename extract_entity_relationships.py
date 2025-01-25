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
    cypher_queries: list[str] = Field(default=[], description="The list of cypher queries to create the knowledge graph")
    root_entity_name: str = Field(default='', description="The name of the root entity. For e.g 'Bob Smith'")

    
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
parser = PydanticOutputParser(pydantic_object=ResumeContentSchema)

template = """
You are a resume parser. You are given a resume and you need to extract entities, relationships and cypher
queries to create a knowledge graph.

output format example:
{{
    'entities': ['Bob Smith', 'John Doe', 'Jane Doe', 'University of California, Los Angeles', 'Google', 'Software Engineer', 'Python', 'Java', 'SQL'],
    'relationships':['HAS_EDUCATION', 'HAS_EXPERIENCE', 'HAS_SKILLS', 'LIVES_IN', 'WORKS_AT'],
    'cypher_queries': ['MERGE (e:Entity {{name: $name}}) RETURN e', 'MERGE (r:Relationship {{name: $name}}) RETURN r'],
    'root_entity_name': 'Bob Smith'
}}

resume:
{text}
"""

prompt = PromptTemplate(template=template)
chain = prompt | llm
response = chain.invoke({"text": full_text})

response_content = response.content if hasattr(response, 'content') else response

try:
    parsed_response = parser.parse(response_content)
except OutputParserException:
    new_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
    parsed_response = new_parser.parse(response_content)
    
entities = parsed_response.entities
relationships = parsed_response.relationships
cypher_queries = parsed_response.cypher_queries
root_entity_name = parsed_response.root_entity_name

print(root_entity_name)

NEO4J_URI = os.getenv('NEO4J_URI_1')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME_1')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD_1')

neo4j_connection = Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

resume_query = """MERGE (r:Resume {name: 'resume'}) RETURN r"""

resume_query_parameters = {
    'name': 'resume'
}

neo4j_connection.write_transaction(resume_query, resume_query_parameters)

file_query = """MERGE (f:File {name: $name}) RETURN f"""

file_query_parameters = {
    'name': 'Bob Smith 1.pdf'
}

neo4j_connection.write_transaction(file_query, file_query_parameters)

relationship_query = """MATCH (r:Resume), (f:File) WHERE r.name = 'resume' AND f.name = 'Bob Smith 1.pdf' CREATE (f)-[:BELONGS_TO]->(r) RETURN r, f"""

neo4j_connection.write_transaction(relationship_query)

for cypher_query in cypher_queries:
    neo4j_connection.write_transaction(cypher_query)

root_entity_query = """MATCH (b { name: $root_entity_name }), (f { name: $file_name }) CREATE (b)-[:HAS_FILE]->(f) RETURN b, f"""

root_entity_query_parameters = {
    'root_entity_name': root_entity_name,
    'file_name': 'Bob Smith 1.pdf'
}

neo4j_connection.write_transaction(root_entity_query, root_entity_query_parameters)

