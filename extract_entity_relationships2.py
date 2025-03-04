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
    entities: list[str] = Field(
        default=[], description="The list of entities in the document"
    )
    relationships: list[str] = Field(
        default=[],
        description="The list of relationships between entities in the document",
    )
    cypher_queries: list[str] = Field(
        default=[],
        description="The list of cypher queries to create the knowledge graph",
    )
    root_entity_name: str = Field(default="", description="The name of the root entity")


def extract_text_from_pdf(pdf_path: str) -> str:
    full_text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            full_text += page.extract_text()
    return full_text


def get_all_nodes_and_relationships():
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
    parser = PydanticOutputParser(pydantic_object=ContentSchema)


    resume_prompt = """
    You are a resume parser. Your primary objective is to analyze the provided resume and extract key components to construct
    a knowledge graph. You will be given the following inputs:

    - A document text (which will be a resume).
    - A list of all previously extracted relationships (all_relationships). For first document, all_relationships will be empty.

    Your task is to extract:
    1. **Entities:** Key names, institutions, concepts, topics, skills, locations, etc.
    2. **Relationships:** The connections between these entities. When determining a relationship, check the provided all_relationships list. 
    If a similar relationship already exists (even if represented with different synonyms), use the existing relationship name.  
    - *For example:*  
        - **HAS_AUTHORED** and **AUTHORED_BY** are considered the same.  
        - **HAS_CONTRIBUTED** and **AUTHORED_BY** are considered the same.  
        - So, if **HAS_AUTHORED** is already in the all_relationships list, do not create **AUTHORED_BY**; instead, use **HAS_AUTHORED**.  
        - Similarly, if **HAS_CONTRIBUTED** exists, do not generate **AUTHORED_BY**; use **HAS_CONTRIBUTED**.
    3. **Cypher Queries:** Generate Cypher queries that logically connect the extracted entities. Each query should:
    - Create or merge nodes for each entity.
    - Create relationships connecting the nodes, ensuring every entity is linked to the designated root entity.
    - Be written in a clear, easy-to-understand manner.
    4. **Root Entity Name:** Identify and assign the main or most representative entity of the document as the root node. Every other entity should be connected directly or indirectly to this root.

    **Important Guidelines:**
    - **Comprehensiveness:** Extract as many entities and relationships as possible. Ensure no relevant piece of information is omitted.
    - **Context Sensitivity:** 
    - For resumes, focus on aspects like Person Name, Education (degrees or institutions), Work Experience (companies and job titles), Skills, Location, Certifications, Awards, etc.
    - **Connection Logic:**  

    - Every node must be connected to the root entity node via an appropriate relationship.
    - For example, if "NED University" is mentioned, it should be connected with a relationship related to education or institution, not one meant for contributions.  
    - Similarly, "Karachi" should be connected with a relationship related to location or residence rather than an institution-related relationship.

    **Output Format:**
    Use the following JSON-like structure as a guide. Do not alter the double curly braces {{ }} as they are required by the Langchain format.
    {{
        'entities': [...],
        'relationships': [...],
        'cypher_queries': [...],
        'root_entity_name': '...'
    }}

    **Detailed Examples:**

    ----------------------------------
    Example 1: Resume Document
    ----------------------------------
    Document:
    "My name is Alice Johnson. I have a Bachelor of Science in Computer Science from the University of Texas. I worked at IBM as a Data Scientist. My skill set includes Python, Machine Learning, and Data Analysis."

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
    Example 2: Resume with Additional Details
    ----------------------------------
    Resume:
    "I am Bob Smith, living in San Francisco. I graduated with a Master's in Data Science from Stanford University, and I have experience at Google as a Machine Learning Engineer. I am proficient in Python, C++, and SQL. I also received the 'Innovator Award' for my contributions to AI research."

    Expected Output:
    {{
        'entities': ['Bob Smith', 'San Francisco', 'Stanford University', 'Google', 'Machine Learning Engineer', 'Python', 'C++', 'SQL', 'Innovator Award', 'AI Research'],
        'relationships': ['LIVES_IN', 'HAS_EDUCATION', 'HAS_EXPERIENCE', 'HAS_SKILLS', 'HAS_AWARDS', 'HAS_RESEARCH'],
        'cypher_queries': [
            "MERGE (person:Entity {{name: 'Bob Smith'}}) RETURN person",
            "MERGE (city:Entity {{name: 'San Francisco'}}) RETURN city",
            "MERGE (university:Entity {{name: 'Stanford University'}}) RETURN university",
            "MERGE (company:Entity {{name: 'Google'}}) RETURN company",
            "MERGE (role:Entity {{name: 'Machine Learning Engineer'}}) RETURN role",
            "MERGE (skill1:Entity {{name: 'Python'}}) RETURN skill1",
            "MERGE (skill2:Entity {{name: 'C++'}}) RETURN skill2",
            "MERGE (skill3:Entity {{name: 'SQL'}}) RETURN skill3",
            "MERGE (award:Entity {{name: 'Innovator Award'}}) RETURN award",
            "MERGE (research:Entity {{name: 'AI Research'}}) RETURN research",
            "MATCH (person:Entity {{name: 'Bob Smith'}}), (city:Entity {{name: 'San Francisco'}}) MERGE (person)-[:LIVES_IN]->(city)",
            "MATCH (person:Entity {{name: 'Bob Smith'}}), (university:Entity {{name: 'Stanford University'}}) MERGE (person)-[:HAS_EDUCATION]->(university)",
            "MATCH (person:Entity {{name: 'Bob Smith'}}), (company:Entity {{name: 'Google'}}) MERGE (person)-[:HAS_EXPERIENCE]->(company)",
            "MATCH (person:Entity {{name: 'Bob Smith'}}), (role:Entity {{name: 'Machine Learning Engineer'}}) MERGE (person)-[:HAS_EXPERIENCE]->(role)",
            "MATCH (person:Entity {{name: 'Bob Smith'}}), (skill1:Entity {{name: 'Python'}}) MERGE (person)-[:HAS_SKILLS]->(skill1)",
            "MATCH (person:Entity {{name: 'Bob Smith'}}), (skill2:Entity {{name: 'C++'}}) MERGE (person)-[:HAS_SKILLS]->(skill2)",
            "MATCH (person:Entity {{name: 'Bob Smith'}}), (skill3:Entity {{name: 'SQL'}}) MERGE (person)-[:HAS_SKILLS]->(skill3)",
            "MATCH (person:Entity {{name: 'Bob Smith'}}), (award:Entity {{name: 'Innovator Award'}}) MERGE (person)-[:HAS_AWARDS]->(award)",
            "MATCH (person:Entity {{name: 'Bob Smith'}}), (research:Entity {{name: 'AI Research'}}) MERGE (person)-[:HAS_RESEARCH]->(research)"
        ],
        'root_entity_name': 'Bob Smith'
    }}
    
        ----------------------------------
    Instructions for Processing the Input:
    Now, using the same approach and output format structure shown in the examples above, parse the following inputs:

    - **Document:** {text}
    - **All Relationships:** {all_relationships}


    Your output should comprehensively list all relevant entities, determine the appropriate relationships (reusing existing relationship names when applicable), and generate clear, logically connected cypher queries that integrate every extracted entity with the chosen root entity. Make sure no entity is left unconnected.

    Remember:  
    - Use clear and descriptive relationship names that reflect the nature of the connection (e.g., HAS_EDUCATION, LIVES_IN, HAS_EXPERIENCE, HAS_ADVANCES, HAS_DEVELOPMENT, HAS_IMPACT, etc.).
    """

    science_article_prompt = """
    You are a science article parser. Your primary objective is to analyze the provided science article and extract key components to construct
    a knowledge graph. You will be given the following inputs:

    - A document text (which will be a science article).
    - A list of all previously extracted relationships (all_relationships). For first document, all_relationships will be empty.

    Your task is to extract:
    1. **Entities:** Key research topics, methodologies, findings, authors, institutions, technologies, datasets, metrics, etc.
    2. **Relationships:** The connections between these entities. When determining a relationship, check the provided all_relationships list. 
    If a similar relationship already exists (even if represented with different synonyms), use the existing relationship name.  
    - *For example:*  
        - **HAS_FINDINGS** and **HAS_RESULTS** are considered the same.  
        - **HAS_METHOD** and **HAS_METHODOLOGY** are considered the same.  
        - So, if **HAS_FINDINGS** is already in the all_relationships list, do not create **HAS_RESULTS**; instead, use **HAS_FINDINGS**.  
        - Similarly, if **HAS_METHOD** exists, do not generate **HAS_METHODOLOGY**; use **HAS_METHOD**.
    3. **Cypher Queries:** Generate Cypher queries that logically connect the extracted entities. Each query should:
    - Create or merge nodes for each entity.
    - Create relationships connecting the nodes, ensuring every entity is linked to the designated root entity.
    - Be written in a clear, easy-to-understand manner.
    4. **Root Entity Name:** Identify and assign the main research topic or paper title as the root node. Every other entity should be connected directly or indirectly to this root.

    **Important Guidelines:**
    - **Comprehensiveness:** Extract as many entities and relationships as possible. Ensure no relevant piece of information is omitted.
    - **Context Sensitivity:** 
    - For science articles, focus on aspects like Research Topics, Methods, Results, Authors, Institutions, Technologies, Datasets, Metrics, etc.
    - **Connection Logic:**  
    - Every node must be connected to the root entity node via an appropriate relationship.
    - For example, if "Machine Learning Algorithm" is mentioned, it should be connected with a relationship related to methodology or technology, not one meant for results.  
    - Similarly, "Stanford University" should be connected with a relationship related to affiliation rather than a methodology-related relationship.

    **Output Format:**
    Use the following JSON-like structure as a guide. Do not alter the double curly braces {{ }} as they are required by the Langchain format.
    {{
        'entities': [...],
        'relationships': [...],
        'cypher_queries': [...],
        'root_entity_name': '...'
    }}

    **Detailed Examples:**

    ----------------------------------
    Example 1: Science Article
    ----------------------------------
    Document:
    "Deep Learning for Climate Change Prediction by Dr. Sarah Chen from MIT. The research introduces a novel neural network architecture for predicting climate patterns. Using historical weather data and advanced GPU processing, the study achieved 95% accuracy in short-term predictions. The findings suggest significant improvements over traditional statistical methods."

    Expected Output:
    {{
        'entities': ['Climate Change Prediction', 'Dr. Sarah Chen', 'MIT', 'Neural Network Architecture', 'Historical Weather Data', 'GPU Processing', '95% Accuracy', 'Statistical Methods'],
        'relationships': ['HAS_AUTHOR', 'HAS_AFFILIATION', 'HAS_METHODOLOGY', 'HAS_DATA', 'HAS_TECHNOLOGY', 'HAS_RESULTS', 'HAS_COMPARISON'],
        'cypher_queries': [
            "MERGE (paper:Entity {{name: 'Climate Change Prediction'}}) RETURN paper",
            "MERGE (author:Entity {{name: 'Dr. Sarah Chen'}}) RETURN author",
            "MERGE (inst:Entity {{name: 'MIT'}}) RETURN inst",
            "MERGE (method:Entity {{name: 'Neural Network Architecture'}}) RETURN method",
            "MERGE (data:Entity {{name: 'Historical Weather Data'}}) RETURN data",
            "MERGE (tech:Entity {{name: 'GPU Processing'}}) RETURN tech",
            "MERGE (result:Entity {{name: '95% Accuracy'}}) RETURN result",
            "MERGE (comp:Entity {{name: 'Statistical Methods'}}) RETURN comp",
            "MATCH (paper:Entity {{name: 'Climate Change Prediction'}}), (author:Entity {{name: 'Dr. Sarah Chen'}}) MERGE (paper)-[:HAS_AUTHOR]->(author)",
            "MATCH (paper:Entity {{name: 'Climate Change Prediction'}}), (inst:Entity {{name: 'MIT'}}) MERGE (paper)-[:HAS_AFFILIATION]->(inst)",
            "MATCH (paper:Entity {{name: 'Climate Change Prediction'}}), (method:Entity {{name: 'Neural Network Architecture'}}) MERGE (paper)-[:HAS_METHODOLOGY]->(method)",
            "MATCH (paper:Entity {{name: 'Climate Change Prediction'}}), (data:Entity {{name: 'Historical Weather Data'}}) MERGE (paper)-[:HAS_DATA]->(data)",
            "MATCH (paper:Entity {{name: 'Climate Change Prediction'}}), (tech:Entity {{name: 'GPU Processing'}}) MERGE (paper)-[:HAS_TECHNOLOGY]->(tech)",
            "MATCH (paper:Entity {{name: 'Climate Change Prediction'}}), (result:Entity {{name: '95% Accuracy'}}) MERGE (paper)-[:HAS_RESULTS]->(result)",
            "MATCH (paper:Entity {{name: 'Climate Change Prediction'}}), (comp:Entity {{name: 'Statistical Methods'}}) MERGE (paper)-[:HAS_COMPARISON]->(comp)"
        ],
        'root_entity_name': 'Climate Change Prediction'
    }}

    ----------------------------------
    Example 2: Science Article with Multiple Authors
    ----------------------------------
    Document:
    "Quantum Computing Breakthrough in Error Correction. Authors: Dr. James Wilson (Google AI) and Prof. Lisa Zhang (Stanford). The team developed a novel quantum error correction protocol that achieves 99.9% fidelity. The research utilized a 50-qubit quantum computer and demonstrated superior performance in maintaining quantum coherence. The implications for quantum computing scalability are significant."

    Expected Output:
    {{
        'entities': ['Quantum Error Correction', 'Dr. James Wilson', 'Prof. Lisa Zhang', 'Google AI', 'Stanford', 'Error Correction Protocol', '99.9% Fidelity', '50-qubit Quantum Computer', 'Quantum Coherence', 'Quantum Computing Scalability'],
        'relationships': ['HAS_AUTHOR', 'HAS_AFFILIATION', 'HAS_METHODOLOGY', 'HAS_RESULTS', 'HAS_EQUIPMENT', 'HAS_IMPACT'],
        'cypher_queries': [
            "MERGE (paper:Entity {{name: 'Quantum Error Correction'}}) RETURN paper",
            "MERGE (author1:Entity {{name: 'Dr. James Wilson'}}) RETURN author1",
            "MERGE (author2:Entity {{name: 'Prof. Lisa Zhang'}}) RETURN author2",
            "MERGE (inst1:Entity {{name: 'Google AI'}}) RETURN inst1",
            "MERGE (inst2:Entity {{name: 'Stanford'}}) RETURN inst2",
            "MERGE (method:Entity {{name: 'Error Correction Protocol'}}) RETURN method",
            "MERGE (result:Entity {{name: '99.9% Fidelity'}}) RETURN result",
            "MERGE (equip:Entity {{name: '50-qubit Quantum Computer'}}) RETURN equip",
            "MERGE (perf:Entity {{name: 'Quantum Coherence'}}) RETURN perf",
            "MERGE (impact:Entity {{name: 'Quantum Computing Scalability'}}) RETURN impact",
            "MATCH (paper:Entity {{name: 'Quantum Error Correction'}}), (author1:Entity {{name: 'Dr. James Wilson'}}) MERGE (paper)-[:HAS_AUTHOR]->(author1)",
            "MATCH (paper:Entity {{name: 'Quantum Error Correction'}}), (author2:Entity {{name: 'Prof. Lisa Zhang'}}) MERGE (paper)-[:HAS_AUTHOR]->(author2)",
            "MATCH (paper:Entity {{name: 'Quantum Error Correction'}}), (inst1:Entity {{name: 'Google AI'}}) MERGE (paper)-[:HAS_AFFILIATION]->(inst1)",
            "MATCH (paper:Entity {{name: 'Quantum Error Correction'}}), (inst2:Entity {{name: 'Stanford'}}) MERGE (paper)-[:HAS_AFFILIATION]->(inst2)",
            "MATCH (paper:Entity {{name: 'Quantum Error Correction'}}), (method:Entity {{name: 'Error Correction Protocol'}}) MERGE (paper)-[:HAS_METHODOLOGY]->(method)",
            "MATCH (paper:Entity {{name: 'Quantum Error Correction'}}), (result:Entity {{name: '99.9% Fidelity'}}) MERGE (paper)-[:HAS_RESULTS]->(result)",
            "MATCH (paper:Entity {{name: 'Quantum Error Correction'}}), (equip:Entity {{name: '50-qubit Quantum Computer'}}) MERGE (paper)-[:HAS_EQUIPMENT]->(equip)",
            "MATCH (paper:Entity {{name: 'Quantum Error Correction'}}), (perf:Entity {{name: 'Quantum Coherence'}}) MERGE (paper)-[:HAS_RESULTS]->(perf)",
            "MATCH (paper:Entity {{name: 'Quantum Error Correction'}}), (impact:Entity {{name: 'Quantum Computing Scalability'}}) MERGE (paper)-[:HAS_IMPACT]->(impact)"
        ],
        'root_entity_name': 'Quantum Error Correction'
    }}

    Instructions for Processing the Input:
    Now, using the same approach and output format structure shown in the examples above, parse the following inputs:

    - **Document:** {text}
    - **All Relationships:** {all_relationships}

    Your output should comprehensively list all relevant entities, determine the appropriate relationships (reusing existing relationship names when applicable), and generate clear, logically connected cypher queries that integrate every extracted entity with the chosen root entity. Make sure no entity is left unconnected.

    Remember:  
    - Use clear and descriptive relationship names that reflect the nature of the connection (e.g., HAS_AUTHOR, HAS_AFFILIATION, HAS_METHODOLOGY, HAS_RESULTS, HAS_IMPACT, etc.).
    """

    technical_document_prompt = """
    You are a technical document parser. Your primary objective is to analyze the provided technical document and extract key components to construct
    a knowledge graph. You will be given the following inputs:

    - A document text (which will be a technical document).
    - A list of all previously extracted relationships (all_relationships). For first document, all_relationships will be empty.

    Your task is to extract:
    1. **Entities:** Key technical concepts, methodologies, technologies, components, specifications, impacts, applications, systems, architectures, etc.
    2. **Relationships:** The connections between these entities. When determining a relationship, check the provided all_relationships list. 
    If a similar relationship already exists (even if represented with different synonyms), use the existing relationship name.  
    - *For example:*  
        - **HAS_COMPONENT** and **CONTAINS_COMPONENT** are considered the same.  
        - **HAS_SPECIFICATION** and **HAS_SPECS** are considered the same.  
        - So, if **HAS_COMPONENT** is already in the all_relationships list, do not create **CONTAINS_COMPONENT**; instead, use **HAS_COMPONENT**.  
        - Similarly, if **HAS_SPECIFICATION** exists, do not generate **HAS_SPECS**; use **HAS_SPECIFICATION**.
    3. **Cypher Queries:** Generate Cypher queries that logically connect the extracted entities. Each query should:
    - Create or merge nodes for each entity.
    - Create relationships connecting the nodes, ensuring every entity is linked to the designated root entity.
    - Be written in a clear, easy-to-understand manner.
    4. **Root Entity Name:** Identify and assign the main technical concept or system as the root node. Every other entity should be connected directly or indirectly to this root.

    **Important Guidelines:**
    - **Comprehensiveness:** Extract as many entities and relationships as possible. Ensure no relevant piece of information is omitted.
    - **Context Sensitivity:** 
    - For technical documents, focus on aspects like Technical Systems, Components, Specifications, Technologies, Architectures, Applications, Impacts, Requirements, etc.
    - **Connection Logic:**  
    - Every node must be connected to the root entity node via an appropriate relationship.
    - For example, if "Processing Unit" is mentioned, it should be connected with a relationship related to components or architecture, not one meant for impacts.  
    - Similarly, "Performance Metrics" should be connected with a relationship related to specifications rather than an application-related relationship.

    **Output Format:**
    Use the following JSON-like structure as a guide. Do not alter the double curly braces {{ }} as they are required by the Langchain format.
    {{
        'entities': [...],
        'relationships': [...],
        'cypher_queries': [...],
        'root_entity_name': '...'
    }}

    **Detailed Examples:**

    ----------------------------------
    Example 1: Technical Document
    ----------------------------------
    Document:
    "The Advanced Robotics Control System (ARCS) is a state-of-the-art platform for industrial automation. The system features a high-performance CPU running at 3.5GHz, integrated motion sensors, and real-time processing capabilities. ARCS has been successfully implemented in manufacturing lines, achieving a 40% increase in production efficiency. The system requires minimal maintenance and operates under standard industrial conditions."

    Expected Output:
    {{
        'entities': ['Advanced Robotics Control System', 'Industrial Automation', 'High-performance CPU', '3.5GHz', 'Motion Sensors', 'Real-time Processing', 'Manufacturing Lines', '40% Production Efficiency', 'Minimal Maintenance', 'Industrial Conditions'],
        'relationships': ['HAS_APPLICATION', 'HAS_COMPONENT', 'HAS_SPECIFICATION', 'HAS_FEATURE', 'HAS_PERFORMANCE', 'HAS_REQUIREMENT'],
        'cypher_queries': [
            "MERGE (system:Entity {{name: 'Advanced Robotics Control System'}}) RETURN system",
            "MERGE (app:Entity {{name: 'Industrial Automation'}}) RETURN app",
            "MERGE (cpu:Entity {{name: 'High-performance CPU'}}) RETURN cpu",
            "MERGE (speed:Entity {{name: '3.5GHz'}}) RETURN speed",
            "MERGE (sensors:Entity {{name: 'Motion Sensors'}}) RETURN sensors",
            "MERGE (processing:Entity {{name: 'Real-time Processing'}}) RETURN processing",
            "MERGE (mfg:Entity {{name: 'Manufacturing Lines'}}) RETURN mfg",
            "MERGE (efficiency:Entity {{name: '40% Production Efficiency'}}) RETURN efficiency",
            "MERGE (maintenance:Entity {{name: 'Minimal Maintenance'}}) RETURN maintenance",
            "MERGE (conditions:Entity {{name: 'Industrial Conditions'}}) RETURN conditions",
            "MATCH (system:Entity {{name: 'Advanced Robotics Control System'}}), (app:Entity {{name: 'Industrial Automation'}}) MERGE (system)-[:HAS_APPLICATION]->(app)",
            "MATCH (system:Entity {{name: 'Advanced Robotics Control System'}}), (cpu:Entity {{name: 'High-performance CPU'}}) MERGE (system)-[:HAS_COMPONENT]->(cpu)",
            "MATCH (system:Entity {{name: 'Advanced Robotics Control System'}}), (speed:Entity {{name: '3.5GHz'}}) MERGE (system)-[:HAS_SPECIFICATION]->(speed)",
            "MATCH (system:Entity {{name: 'Advanced Robotics Control System'}}), (sensors:Entity {{name: 'Motion Sensors'}}) MERGE (system)-[:HAS_COMPONENT]->(sensors)",
            "MATCH (system:Entity {{name: 'Advanced Robotics Control System'}}), (processing:Entity {{name: 'Real-time Processing'}}) MERGE (system)-[:HAS_FEATURE]->(processing)",
            "MATCH (system:Entity {{name: 'Advanced Robotics Control System'}}), (mfg:Entity {{name: 'Manufacturing Lines'}}) MERGE (system)-[:HAS_APPLICATION]->(mfg)",
            "MATCH (system:Entity {{name: 'Advanced Robotics Control System'}}), (efficiency:Entity {{name: '40% Production Efficiency'}}) MERGE (system)-[:HAS_PERFORMANCE]->(efficiency)",
            "MATCH (system:Entity {{name: 'Advanced Robotics Control System'}}), (maintenance:Entity {{name: 'Minimal Maintenance'}}) MERGE (system)-[:HAS_REQUIREMENT]->(maintenance)",
            "MATCH (system:Entity {{name: 'Advanced Robotics Control System'}}), (conditions:Entity {{name: 'Industrial Conditions'}}) MERGE (system)-[:HAS_REQUIREMENT]->(conditions)"
        ],
        'root_entity_name': 'Advanced Robotics Control System'
    }}

    ----------------------------------
    Example 2: Technical Document with Architecture Details
    ----------------------------------
    Document:
    "The Cloud-Native Security Platform (CNSP) implements a microservices architecture for enhanced cybersecurity. The system consists of containerized security modules, a distributed database, and AI-powered threat detection. Key features include real-time monitoring, automated response capabilities, and integration with major cloud providers. Testing shows 99.99% uptime and sub-millisecond response times."

    Expected Output:
    {{
        'entities': ['Cloud-Native Security Platform', 'Microservices Architecture', 'Containerized Security Modules', 'Distributed Database', 'AI-powered Threat Detection', 'Real-time Monitoring', 'Automated Response', 'Cloud Provider Integration', '99.99% Uptime', 'Sub-millisecond Response Times'],
        'relationships': ['HAS_ARCHITECTURE', 'HAS_COMPONENT', 'HAS_FEATURE', 'HAS_CAPABILITY', 'HAS_INTEGRATION', 'HAS_PERFORMANCE'],
        'cypher_queries': [
            "MERGE (platform:Entity {{name: 'Cloud-Native Security Platform'}}) RETURN platform",
            "MERGE (arch:Entity {{name: 'Microservices Architecture'}}) RETURN arch",
            "MERGE (modules:Entity {{name: 'Containerized Security Modules'}}) RETURN modules",
            "MERGE (db:Entity {{name: 'Distributed Database'}}) RETURN db",
            "MERGE (ai:Entity {{name: 'AI-powered Threat Detection'}}) RETURN ai",
            "MERGE (monitoring:Entity {{name: 'Real-time Monitoring'}}) RETURN monitoring",
            "MERGE (response:Entity {{name: 'Automated Response'}}) RETURN response",
            "MERGE (integration:Entity {{name: 'Cloud Provider Integration'}}) RETURN integration",
            "MERGE (uptime:Entity {{name: '99.99% Uptime'}}) RETURN uptime",
            "MERGE (latency:Entity {{name: 'Sub-millisecond Response Times'}}) RETURN latency",
            "MATCH (platform:Entity {{name: 'Cloud-Native Security Platform'}}), (arch:Entity {{name: 'Microservices Architecture'}}) MERGE (platform)-[:HAS_ARCHITECTURE]->(arch)",
            "MATCH (platform:Entity {{name: 'Cloud-Native Security Platform'}}), (modules:Entity {{name: 'Containerized Security Modules'}}) MERGE (platform)-[:HAS_COMPONENT]->(modules)",
            "MATCH (platform:Entity {{name: 'Cloud-Native Security Platform'}}), (db:Entity {{name: 'Distributed Database'}}) MERGE (platform)-[:HAS_COMPONENT]->(db)",
            "MATCH (platform:Entity {{name: 'Cloud-Native Security Platform'}}), (ai:Entity {{name: 'AI-powered Threat Detection'}}) MERGE (platform)-[:HAS_FEATURE]->(ai)",
            "MATCH (platform:Entity {{name: 'Cloud-Native Security Platform'}}), (monitoring:Entity {{name: 'Real-time Monitoring'}}) MERGE (platform)-[:HAS_CAPABILITY]->(monitoring)",
            "MATCH (platform:Entity {{name: 'Cloud-Native Security Platform'}}), (response:Entity {{name: 'Automated Response'}}) MERGE (platform)-[:HAS_CAPABILITY]->(response)",
            "MATCH (platform:Entity {{name: 'Cloud-Native Security Platform'}}), (integration:Entity {{name: 'Cloud Provider Integration'}}) MERGE (platform)-[:HAS_INTEGRATION]->(integration)",
            "MATCH (platform:Entity {{name: 'Cloud-Native Security Platform'}}), (uptime:Entity {{name: '99.99% Uptime'}}) MERGE (platform)-[:HAS_PERFORMANCE]->(uptime)",
            "MATCH (platform:Entity {{name: 'Cloud-Native Security Platform'}}), (latency:Entity {{name: 'Sub-millisecond Response Times'}}) MERGE (platform)-[:HAS_PERFORMANCE]->(latency)"
        ],
        'root_entity_name': 'Cloud-Native Security Platform'
    }}

    Instructions for Processing the Input:
    Now, using the same approach and output format structure shown in the examples above, parse the following inputs:

    - **Document:** {text}
    - **All Relationships:** {all_relationships}

    Your output should comprehensively list all relevant entities, determine the appropriate relationships (reusing existing relationship names when applicable), and generate clear, logically connected cypher queries that integrate every extracted entity with the chosen root entity. Make sure no entity is left unconnected.

    Remember:  
    - Use clear and descriptive relationship names that reflect the nature of the connection (e.g., HAS_COMPONENT, HAS_FEATURE, HAS_SPECIFICATION, HAS_PERFORMANCE, HAS_REQUIREMENT, etc.).
    """

    # Select the appropriate prompt template based on the document class
    if doc_class == DocClass.RESUME.value:
        template = resume_prompt
    elif doc_class == DocClass.SCIENCE_ARTICLE.value:
        template = science_article_prompt
    elif doc_class == DocClass.TECHNICAL_DOCUMENT.value:
        template = technical_document_prompt
    else:
        raise ValueError("Invalid document class provided.")

    prompt = PromptTemplate(template=template)
    chain = prompt | llm

    response = chain.invoke({"text": full_text, "all_relationships": all_relationships})
    response_content = response.content if hasattr(response, "content") else response

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

    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

    neo4j_connection = Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

    query = f"""MERGE (r:{doc_class} {{name: '{doc_class.lower()}'}}) RETURN r"""
    neo4j_connection.write_transaction(query)

    file_name = os.path.basename(pdf_path)

    file_query = """MERGE (f:File {name: $file_name}) RETURN f"""
    file_query_params = {"file_name": file_name}
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
        "root_entity_name": root_entity_name,
        "file_name": file_name,
    }
    neo4j_connection.write_transaction(root_entity_query, root_entity_query_params)

    print(f"Finished processing {file_name} into the Neo4j graph.")


if __name__ == "__main__":
    pdf_path = "docs/Muhammad Faris Khan CV.pdf"
    doc_class = (
        DocClass.RESUME.value
    )  # Change this to SCIENCE_ARTICLE or TECHNICAL_DOCUMENT as needed
    process_document(pdf_path, doc_class)
