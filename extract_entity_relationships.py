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

    Use the following output format as a guide:
    {{
        'entities': [...],
        'relationships': [...],
        'cypher_queries': [...],
        'root_entity_name': '...'
    }}

    Below are examples demonstrating the expected behavior:

    Example 1:
    ----------------------------------
    Resume:
    "My name is Alice Johnson. I have a Bachelor of Science in Computer Science from the University of Texas. 
    I worked at IBM as a Data Scientist. My skill set includes Python, Machine Learning, and Data Analysis."
    Expected Output:
    {{
        'entities': ['Alice Johnson', 'University of Texas', 'IBM', 'Data Scientist', 'Python', 'Machine Learning', 'Data Analysis'],
        'relationships': ['HAS_EDUCATION', 'HAS_EXPERIENCE', 'HAS_SKILLS'],
        'cypher_queries': [
            "MERGE (person:Entity {{name: 'Alice Johnson'}}) RETURN person",
            "MERGE (school:Entity {{name: 'University of Texas'}}) RETURN school",
            "MERGE (company:Entity {{name: 'IBM'}}) RETURN company",
            "MERGE (role:Entity {{name: 'Data Scientist'}}) RETURN role",
            "MERGE (skill1:Entity {{name: 'Python'}}) RETURN skill1",
            "MERGE (skill2:Entity {{name: 'Machine Learning'}}) RETURN skill2",
            "MERGE (skill3:Entity {{name: 'Data Analysis'}}) RETURN skill3",
            "MATCH (person:Entity {{name: 'Alice Johnson'}}), (school:Entity {{name: 'University of Texas'}}) MERGE (person)-[:HAS_EDUCATION]->(school)",
            "MATCH (person:Entity {{name: 'Alice Johnson'}}), (company:Entity {{name: 'IBM'}}) MERGE (person)-[:HAS_EXPERIENCE]->(company)",
            "MATCH (person:Entity {{name: 'Alice Johnson'}}), (role:Entity {{name: 'Data Scientist'}}) MERGE (person)-[:HAS_EXPERIENCE]->(role)",
            "MATCH (person:Entity {{name: 'Alice Johnson'}}), (skill1:Entity {{name: 'Python'}}) MERGE (person)-[:HAS_SKILLS]->(skill1)",
            "MATCH (person:Entity {{name: 'Alice Johnson'}}), (skill2:Entity {{name: 'Machine Learning'}}) MERGE (person)-[:HAS_SKILLS]->(skill2)",
            "MATCH (person:Entity {{name: 'Alice Johnson'}}), (skill3:Entity {{name: 'Data Analysis'}}) MERGE (person)-[:HAS_SKILLS]->(skill3)"
        ],
        'root_entity_name': 'Alice Johnson'
    }}
    ----------------------------------

    Example 2:
    ----------------------------------
    Resume:
    "I am Bob Smith, living in San Francisco. I graduated with a Master's in Data Science from Stanford University, 
    and I have experience at Google as a Machine Learning Engineer. I am proficient in Python, C++, and SQL."
    Expected Output:
    {{
        'entities': ['Bob Smith', 'San Francisco', 'Stanford University', 'Google', 'Machine Learning Engineer', 'Python', 'C++', 'SQL'],
        'relationships': ['LIVES_IN', 'HAS_EDUCATION', 'HAS_EXPERIENCE', 'HAS_SKILLS'],
        'cypher_queries': [
            "MERGE (person:Entity {{name: 'Bob Smith'}}) RETURN person",
            "MERGE (city:Entity {{name: 'San Francisco'}}) RETURN city",
            "MERGE (university:Entity {{name: 'Stanford University'}}) RETURN university",
            "MERGE (company:Entity {{name: 'Google'}}) RETURN company",
            "MERGE (role:Entity {{name: 'Machine Learning Engineer'}}) RETURN role",
            "MERGE (skill1:Entity {{name: 'Python'}}) RETURN skill1",
            "MERGE (skill2:Entity {{name: 'C++'}}) RETURN skill2",
            "MERGE (skill3:Entity {{name: 'SQL'}}) RETURN skill3",
            "MATCH (person:Entity {{name: 'Bob Smith'}}), (city:Entity {{name: 'San Francisco'}}) MERGE (person)-[:LIVES_IN]->(city)",
            "MATCH (person:Entity {{name: 'Bob Smith'}}), (university:Entity {{name: 'Stanford University'}}) MERGE (person)-[:HAS_EDUCATION]->(university)",
            "MATCH (person:Entity {{name: 'Bob Smith'}}), (company:Entity {{name: 'Google'}}) MERGE (person)-[:HAS_EXPERIENCE]->(company)",
            "MATCH (person:Entity {{name: 'Bob Smith'}}), (role:Entity {{name: 'Machine Learning Engineer'}}) MERGE (person)-[:HAS_EXPERIENCE]->(role)",
            "MATCH (person:Entity {{name: 'Bob Smith'}}), (skill1:Entity {{name: 'Python'}}) MERGE (person)-[:HAS_SKILLS]->(skill1)",
            "MATCH (person:Entity {{name: 'Bob Smith'}}), (skill2:Entity {{name: 'C++'}}) MERGE (person)-[:HAS_SKILLS]->(skill2)",
            "MATCH (person:Entity {{name: 'Bob Smith'}}), (skill3:Entity {{name: 'SQL'}}) MERGE (person)-[:HAS_SKILLS]->(skill3)"
        ],
        'root_entity_name': 'Bob Smith'
    }}
    ----------------------------------

    Now, using the same approach and the same output format structure, parse the following resume text into its relevant entities, relationships, 
    and cypher queries. Make sure to assign the 'root_entity_name' to the candidate's name. 
    Focus on the following categories (if present) to derive entities and relationships: 
    - Person Name
    - Education (Universities or Degrees)
    - Work Experience (Companies and Job Titles)
    - Skills (Technical or Domain-Specific)
    - Location or Residence
    - Others (Certifications, Awards, etc.)

    Resume:
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
    pdf_path = "docs/Hasnain Ali Resume.pdf"
    process_resume(pdf_path)
