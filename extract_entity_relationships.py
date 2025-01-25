import os
import PyPDF2

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field

from main import Neo4jConnection

class ResumeContentSchema(BaseModel):
    entities: list[str] = Field(default=[], description="The list of entities in the resume")
    relationships: list[str] = Field(default=[], description="The list of relationships between entities in the resume")
    cypher_queries: list[str] = Field(default=[], description="The list of cypher queries to create the knowledge graph")
    root_entity_name: str = Field(default='', description="The name of the root entity. For e.g 'Bob Smith'")

def extract_text_from_pdf(pdf_path: str) -> str:
    full_text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            full_text += page.extract_text()
    return full_text

def process_resume(pdf_path: str):
    full_text = extract_text_from_pdf(pdf_path)
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    parser = PydanticOutputParser(pydantic_object=ResumeContentSchema)
    
    template = """
    You are a resume parser. You are given a resume and you need to extract entities, 
    relationships, and cypher queries to create a knowledge graph.

    output format example:
    {{
        'entities': ['Bob Smith', 'John Doe', 'Jane Doe', 'University of California, Los Angeles', 'Google', 'Software Engineer', 'Python', 'Java', 'SQL'],
        'relationships': ['HAS_EDUCATION', 'HAS_EXPERIENCE', 'HAS_SKILLS', 'LIVES_IN', 'WORKS_AT'],
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
    
    print("Root entity name extracted:", root_entity_name)
    print("Entities extracted:", entities)
    print("Relationships extracted:", relationships)
    print("Cypher queries extracted:", cypher_queries)

    NEO4J_URI = os.getenv('NEO4J_URI_1')
    NEO4J_USERNAME = os.getenv('NEO4J_USERNAME_1')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD_1')

    neo4j_connection = Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

    resume_query = """MERGE (r:Resume {name: 'resume'}) RETURN r"""
    neo4j_connection.write_transaction(resume_query)

    file_name = os.path.basename(pdf_path)
    
    file_query = """MERGE (f:File {name: $file_name}) RETURN f"""
    file_query_params = {'file_name': file_name}
    neo4j_connection.write_transaction(file_query, file_query_params)

    belongs_query = """
        MATCH (r:Resume {name: 'resume'}), (f:File {name: $file_name})
        MERGE (f)-[:BELONGS_TO]->(r) 
        RETURN r, f
    """
    neo4j_connection.write_transaction(belongs_query, file_query_params)

    for cypher_query in cypher_queries:
        neo4j_connection.write_transaction(cypher_query)

    root_entity_query = """
        MATCH (b { name: $root_entity_name }), (f { name: $file_name })
        MERGE (b)-[:HAS_FILE]->(f)
        RETURN b, f
    """
    root_entity_query_params = {
        'root_entity_name': root_entity_name,
        'file_name': file_name
    }
    neo4j_connection.write_transaction(root_entity_query, root_entity_query_params)

    print(f"Finished processing {file_name} into the Neo4j graph.")

if __name__ == "__main__":
    pdf_path = "docs/George Kim 1.pdf"
    process_resume(pdf_path)
