import os
import PyPDF2

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field

from main import Neo4jConnection
from neo4j import GraphDatabase
from enum import Enum

class DocClass(Enum):
    RESUME = "resume"
    SCIENCE_ARTICLE = "science_article"
    TECHNICAL_DOCUMENT = "technical_document"

class ContentSchema(BaseModel):
    entities: list[str] = Field(default=[], description="The list of entities in the resume")
    relationships: list[str] = Field(default=[], description="The list of relationships between entities in the resume")
    cypher_queries: list[str] = Field(default=[], description="The list of cypher queries to create the knowledge graph")
    root_entity_name: str = Field(default='', description="The name of the root entity. For e.g 'Bob Smith', 'Robotics Article'")

def extract_text_from_pdf(pdf_path: str) -> str:
    full_text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            full_text += page.extract_text()
    return full_text
    
def get_all_nodes_and_relationships():
    query = """MATCH (n) OPTIONAL MATCH (n)-[r]-() RETURN DISTINCT n, r"""
    
    NEO4J_URI = os.getenv('NEO4J_URI_1')
    NEO4J_USERNAME = os.getenv('NEO4J_USERNAME_1')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD_1')
    
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    
    nodes = []
    relationships = []
    
    try:
        with driver.session() as session:
            result = session.run(query)

            for record in result:
                node = record["n"]
                rel = record["r"]
                nodes.append(node._properties['name'])
                if rel is not None:
                    relationships.append(rel.type)
    finally:
        driver.close()
    
    return nodes, relationships

    
def process_document(pdf_path: str, doc_class: str):
    
    full_text = extract_text_from_pdf(pdf_path)
    
    all_nodes, all_relationships = get_all_nodes_and_relationships()
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    parser = PydanticOutputParser(pydantic_object=ContentSchema)
    
    template = """
    You are a document parser. You are given a document, class of the document and all relationships list.
    You need to extract entities, relationships, and cypher queries to create a knowledge graph. 
    
    All the previously extracted relationships are provided, so when creating new relationships,
    make sure that relationship name ,if not present in the all_relationships list. Then only create the relationship with the new name.
    
    for e.g HAS_AUTHORED and AUTHORED_BY are the same relationship, so if HAS_AUTHORED is already in the list of all_relationships, 
    then you should not create AUTHORED_BY, instead you should use HAS_AUTHORED.

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
    Document:
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
        'relationships': ['LIVES_IN', 'HAS_EDUCATION', 'HAS_EXPERIENCE', 'HAS_SKILLS', 'HAS_PUBLICATIONS'],
        'cypher_queries': [
            "MERGE (person:Entity {{name: 'Bob Smith'}}) RETURN person",
            "MERGE (city:Entity {{name: 'San Francisco'}}) RETURN city",
            "MERGE (university:Entity {{name: 'Stanford University'}}) RETURN university",
            "MERGE (company:Entity {{name: 'Google'}}) RETURN company",
            "MERGE (role:Entity {{name: 'Machine Learning Engineer'}}) RETURN role",
            "MERGE (skill1:Entity {{name: 'Python'}}) RETURN skill1",
            "MERGE (skill2:Entity {{name: 'C++'}}) RETURN skill2",
            "MERGE (skill3:Entity {{name: 'SQL'}}) RETURN skill3",
            "MERGE (publication:Entity {{name: 'Machine Learning Engineer'}}) RETURN publication",
            "MATCH (person:Entity {{name: 'Bob Smith'}}), (city:Entity {{name: 'San Francisco'}}) MERGE (person)-[:LIVES_IN]->(city)",
            "MATCH (person:Entity {{name: 'Bob Smith'}}), (university:Entity {{name: 'Stanford University'}}) MERGE (person)-[:HAS_EDUCATION]->(university)",
            "MATCH (person:Entity {{name: 'Bob Smith'}}), (company:Entity {{name: 'Google'}}) MERGE (person)-[:HAS_EXPERIENCE]->(company)",
            "MATCH (person:Entity {{name: 'Bob Smith'}}), (role:Entity {{name: 'Machine Learning Engineer'}}) MERGE (person)-[:HAS_EXPERIENCE]->(role)",
            "MATCH (person:Entity {{name: 'Bob Smith'}}), (skill1:Entity {{name: 'Python'}}) MERGE (person)-[:HAS_SKILLS]->(skill1)",
            "MATCH (person:Entity {{name: 'Bob Smith'}}), (skill2:Entity {{name: 'C++'}}) MERGE (person)-[:HAS_SKILLS]->(skill2)",
            "MATCH (person:Entity {{name: 'Bob Smith'}}), (skill3:Entity {{name: 'SQL'}}) MERGE (person)-[:HAS_SKILLS]->(skill3)",
            "MATCH (person:Entity {{name: 'Bob Smith'}}), (publication:Entity {{name: 'Machine Learning Engineer'}}) MERGE (person)-[:HAS_PUBLICATIONS]->(publication)"
        ],
        'root_entity_name': 'Bob Smith'
    }}
    
    Example 3:
    Science Article:
    "The article discusses the latest advancements in robotics, including the development of a new robotic arm that can perform complex tasks."
    Expected Output:
    {{
        'entities': ['Robotics', 'Robotic Arm', 'Complex Tasks'],
        'relationships': ['HAS_ADVANCES', 'HAS_DEVELOPMENT'],
        'cypher_queries': [
            "MERGE (article:Entity {{name: 'Robotics Article'}}) RETURN article",
            "MERGE (advancement:Entity {{name: 'Robotics'}}) RETURN advancement",
            "MERGE (arm:Entity {{name: 'Robotic Arm'}}) RETURN arm",
            "MERGE (task:Entity {{name: 'Complex Tasks'}}) RETURN task"
        ],
        'root_entity_name': 'Robotics'
    }}
    
    Example 4:
    Technical Document:
    "This Handbook is a comprehensive guide to the latest advancements in AI, including the development of a new AI model that can perform complex tasks.
    AI is the future of the world."
    Expected Output:
    {{
        'entities': ['AI', 'AI Model', 'Complex Tasks', 'Future of the World'],
        'relationships': ['HAS_ADVANCES', 'HAS_DEVELOPMENT', 'HAS_IMPACT'],
        'cypher_queries': [
            "MERGE (document:Entity {{name: 'Handbook'}}) RETURN document",
            "MERGE (advancement:Entity {{name: 'AI'}}) RETURN advancement",
            "MERGE (model:Entity {{name: 'AI Model'}}) RETURN model",
            "MERGE (impact:Entity {{name: 'Future of the World'}}) RETURN impact"
        ],
        'root_entity_name': 'AI'
    }}
    
    ----------------------------------

    Now, using the same approach and the same output format structure, parse the following document text into its relevant entities, relationships, 
    and cypher queries. Make sure to assign the 'root_entity_name' to the main node that is the most representative of the document. 
    Focus on the following categories (if present) to derive entities and relationships: 
    
    FOR RESUME:
    - Person Name
    - Education (Universities or Degrees)
    - Work Experience (Companies and Job Titles)
    - Skills (Technical or Domain-Specific)
    - Location or Residence
    - Others (Certifications, Awards, etc.)
    
    FOR SCIENCE ARTICLE:
    - Topic
    - Subject
    - Research
    - Methodology
    - Results
    - Conclusion

    FOR TECHNICAL DOCUMENT:
    - Topic
    - Technical Details
    - How it works
    - How it is used
    - Impact

    Document:
    {text}
    
    Document Class:
    {doc_class}
    
    All Relationships:
    {all_relationships}
    """

    
    prompt = PromptTemplate(template=template)
    chain = prompt | llm
    
    response = chain.invoke({"text": full_text, "doc_class": doc_class, "all_relationships": all_relationships})
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

    query = f"""MERGE (r:{doc_class} {{name: '{doc_class.lower()}'}}) RETURN r"""
    neo4j_connection.write_transaction(query)

    file_name = os.path.basename(pdf_path)
    
    file_query = """MERGE (f:File {name: $file_name}) RETURN f"""
    file_query_params = {'file_name': file_name}
    neo4j_connection.write_transaction(file_query, file_query_params)

    belongs_query = f"""
        MATCH (r:{doc_class} {{ name: '{doc_class.lower()}' }}), (f:File {{ name: $file_name }})
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
    pdf_path = "docs/Motor_Parametric_Calculations_for_Robot.pdf" 
    doc_class = DocClass.SCIENCE_ARTICLE.value
    process_document(pdf_path, doc_class)
