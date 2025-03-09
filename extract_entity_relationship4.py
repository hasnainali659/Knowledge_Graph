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

from dotenv import load_dotenv

load_dotenv()

class DocClass(Enum):
    RESUME = "resume"
    SCIENCE_ARTICLE = "science_article"
    TECHNICAL_DOCUMENT = "technical_document"

class ContentSchema(BaseModel):
    root_entity_name: str = Field(
        default="", description="The name of the root entity (the person)."
    )

    skills: list[str] = Field(default=[])
    experience: list[str] = Field(default=[])
    education: list[str] = Field(default=[])
    certifications: list[str] = Field(default=[])
    publications: list[str] = Field(default=[])
    personal_details: list[str] = Field(default=[])

def extract_text_from_pdf(pdf_path: str) -> str:
    full_text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            full_text += page.extract_text()
    return full_text

def get_all_nodes_and_relationships():
    """
    Example function to see what relationships exist.
    You may not need this if you're not reusing relationships.
    """
    query = """MATCH (n) OPTIONAL MATCH (n)-[r]-() RETURN DISTINCT n, r"""

    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    nodes = []
    relationships = []
    try:
        with driver.session() as session:
            result = session.run(query)
            for record in result:
                node = record["n"]
                rel = record["r"]

                if node and "name" in node._properties:
                    nodes.append(node._properties["name"])
                if rel is not None:
                    relationships.append(rel.type)
    finally:
        driver.close()

    return nodes, relationships

def process_document(pdf_path: str, doc_class: str):

    full_text = extract_text_from_pdf(pdf_path)
    
    all_nodes, all_relationships = get_all_nodes_and_relationships()

    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    prompt_text = f"""
    You are a resume parser. Extract the person's name as root_entity_name,
    plus a list of skills, experience, education, certifications, publications,
    and personal details from the resume below:

    RESUME TEXT:
    {full_text}

    Return JSON with fields:
    root_entity_name, skills, experience, education, certifications,
    publications, personal_details
    """

    parser = PydanticOutputParser(pydantic_object=ContentSchema)
    prompt = PromptTemplate(template=prompt_text)
    chain = prompt | llm
    response = chain.invoke({})
    response_content = response.content if hasattr(response, "content") else response

    try:
        parsed_response = parser.parse(response_content)
    except OutputParserException:
        new_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
        parsed_response = new_parser.parse(response_content)

    root_entity_name = parsed_response.root_entity_name
    extracted_skills = parsed_response.skills
    extracted_experience = parsed_response.experience
    extracted_education = parsed_response.education
    extracted_certifications = parsed_response.certifications
    extracted_publications = parsed_response.publications
    extracted_personal_details = parsed_response.personal_details

    print("Root entity name extracted:", root_entity_name)
    print("Skills extracted:", extracted_skills)
    print("Experience extracted:", extracted_experience)
    print("Education extracted:", extracted_education)
    print("Certifications extracted:", extracted_certifications)
    print("Publications extracted:", extracted_publications)
    print("Personal details extracted:", extracted_personal_details)

    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

    neo4j_connection = Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    
    query = f"""
    MERGE (r:RESUME {{ name: 'resume' }})
    RETURN r
    """
    neo4j_connection.write_transaction(query)

    file_name = os.path.basename(pdf_path)
    query = """
    MERGE (f:FILE {name: $file_name, root_entity_name: $root_entity_name})
    RETURN f
    """
    neo4j_connection.write_transaction(query, {"file_name": file_name, "root_entity_name": root_entity_name})

    query = """
    MATCH (r:RESUME { name: 'resume' }), (f:FILE { name: $file_name })
    MERGE (r)-[:HAS_FILE]->(f)
    RETURN r, f
    """
    neo4j_connection.write_transaction(query, {"file_name": file_name})

    categories = [
        ("SKILLS", "HAS_SKILLS", extracted_skills),
        ("EXPERIENCE", "HAS_EXPERIENCE", extracted_experience),
        ("EDUCATION", "HAS_EDUCATION", extracted_education),
        ("CERTIFICATIONS", "HAS_CERTIFICATIONS", extracted_certifications),
        ("PUBLICATIONS", "HAS_PUBLICATIONS", extracted_publications),
        ("PERSONAL_DETAILS", "HAS_PERSONAL_DETAILS", extracted_personal_details)
    ]

    for (cat_label, cat_rel, items_list) in categories:
        cat_node_name = cat_label.lower()
        query = f"""
        MERGE (c:{cat_label} {{ name: $cat_node_name }})
        RETURN c
        """
        neo4j_connection.write_transaction(query, {"cat_node_name": cat_node_name})

        query = f"""
        MATCH (f:FILE {{ name: $file_name }}),
              (c:{cat_label} {{ name: $cat_node_name }})
        MERGE (f)-[:{cat_rel}]->(c)
        RETURN f, c
        """
        neo4j_connection.write_transaction(
            query,
            {"file_name": file_name, "cat_node_name": cat_node_name},
        )

        for item in items_list:
            query = """
            MERGE (i:ITEM { name: $item_name })
            RETURN i
            """
            neo4j_connection.write_transaction(query, {"item_name": item})

            query = f"""
            MATCH (c:{cat_label} {{ name: $cat_node_name }}),
                  (i:ITEM {{ name: $item_name }})
            MERGE (c)-[:HAS_VALUE]->(i)
            RETURN c, i
            """
            neo4j_connection.write_transaction(
                query,
                {"cat_node_name": cat_node_name, "item_name": item},
            )

    print(f"Finished processing {file_name} into the Neo4j graph with a two-level structure.")

if __name__ == "__main__":
    pdf_path = "docs/Hasnain Ali Resume.pdf"
    doc_class = DocClass.RESUME.value  # Or SCIENCE_ARTICLE, TECHNICAL_DOCUMENT
    process_document(pdf_path, doc_class)
