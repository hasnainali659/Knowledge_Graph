from neo4j import GraphDatabase
from datetime import datetime
from typing import Optional, Dict, Any

from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field

from typing import Union
import os
import PyPDF2
from enum import Enum

class ResumeContentSchema(BaseModel):
    header: str = Field(default="", description="The header of the resume")
    education: str = Field(default="", description="The education of the resume")
    experience: str = Field(default="", description="The experience of the resume")
    skills: str = Field(default="", description="The skills of the resume")

class DocumentClass(Enum):
    RESUME = "RESUME"

class Neo4jConnection:
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """
        Initialize Neo4j connection
        
        Args:
            uri (str): Neo4j URI (e.g., "bolt://localhost:7687")
            username (str): Neo4j username
            password (str): Neo4j password
            database (str): Database name (default is "neo4j")
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = None
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            print("Connection to Neo4j DB successful")
        except Exception as e:
            print(f"Failed to connect to Neo4j DB: {e}")
    
    def close(self):
        """
        Close the Neo4j driver connection
        """
        if self.driver is not None:
            self.driver.close()
            print("Neo4j connection closed")
    
    def query(self, query: str, parameters: dict = None) -> Union[list, None]:
        """
        Execute a Cypher query
        
        Args:
            query (str): Cypher query
            parameters (dict): Query parameters (optional)
            
        Returns:
            list: Query results or None if error occurs
        """
        assert self.driver is not None, "Driver not initialized!"
        session = None
        response = None
        
        try:
            session = self.driver.session(database=self.database)
            response = list(session.run(query, parameters or {}))
        except Exception as e:
            print(f"Query failed: {e}")
        finally:
            if session is not None:
                session.close()
        
        return response

    def write_transaction(self, query: str, parameters: dict = None):
        """
        Execute a write transaction
        
        Args:
            query (str): Cypher query for writing data
            parameters (dict): Query parameters (optional)
        """
        assert self.driver is not None, "Driver not initialized!"
        session = None
        
        try:
            session = self.driver.session(database=self.database)
            session.execute_write(
                lambda tx: tx.run(query, parameters or {})
            )
        except Exception as e:
            print(f"Write transaction failed: {e}")
        finally:
            if session is not None:
                session.close()


class PDFDocumentReader:
    def __init__(self, document_class: DocumentClass = DocumentClass.RESUME):
        """
        Initialize PDF Document Reader with a specific document class
        
        Args:
            document_class (DocumentClass): Type of document being processed
        """
        self.document_class = document_class
        self.metadata: Dict[str, Any] = {}
    
    def read_pdf(self, file_path: str) -> Optional[dict]:
        """
        Read a PDF file and extract text based on document class
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            Optional[dict]: Dictionary containing extracted text and metadata
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                self.metadata = {
                    'document_class': self.document_class.value,
                    'num_pages': len(pdf_reader.pages),
                    'file_name': os.path.basename(file_path),
                    'file_size': os.path.getsize(file_path)
                }
                
                if self.document_class == DocumentClass.RESUME:
                    return self._process_resume_using_llm(pdf_reader)
                
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None
        except PyPDF2.PdfReadError as e:
            print(f"Error: Invalid or corrupted PDF file - {e}")
            return None
        except Exception as e:
            print(f"Error: An unexpected error occurred - {e}")
            return None

    def _process_resume(self, pdf_reader: PyPDF2.PdfReader) -> dict:
        """
        Process PDF specifically as a resume
        
        Args:
            pdf_reader (PyPDF2.PdfReader): PDF reader object
            
        Returns:
            dict: Processed resume data
        """
        resume_data = {
            'metadata': self.metadata,
            'content': {
                'full_text': '',
                'sections': {
                    'header': '',
                    'education': '',
                    'experience': '',
                    'skills': ''
                }
            }
        }
        
        full_text = ''
        for page in pdf_reader.pages:
            text = page.extract_text()
            full_text += text + '\n\n'
        
        resume_data['content']['full_text'] = full_text
        
        text_lower = full_text.lower()
        
        if 'education' in text_lower:
            resume_data['content']['sections']['education'] = self._extract_section(full_text, 'education')
            
        if 'experience' in text_lower:
            resume_data['content']['sections']['experience'] = self._extract_section(full_text, 'experience')
            
        if 'skills' in text_lower:
            resume_data['content']['sections']['skills'] = self._extract_section(full_text, 'skills')
        
        return resume_data
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """
        Helper method to extract specific sections from text
        
        Args:
            text (str): Full text content
            section_name (str): Name of the section to extract
            
        Returns:
            str: Extracted section content
        """

        text_lower = text.lower()
        start_idx = text_lower.find(section_name.lower())
        if start_idx == -1:
            return ''
            
        next_section_idx = float('inf')
        for section in ['education', 'experience', 'skills', 'references']:
            if section == section_name:
                continue
            idx = text_lower.find(section, start_idx + len(section_name))
            if idx != -1 and idx < next_section_idx:
                next_section_idx = idx
        
        if next_section_idx == float('inf'):
            return text[start_idx:].strip()
        return text[start_idx:next_section_idx].strip()
    
    def _process_resume_using_llm(self, pdf_reader: PyPDF2.PdfReader) -> dict:
        
        resume_data = {
            'metadata': self.metadata,
            'content': {
                'full_text': '',
                'sections': {
                    'header': '',
                    'education': '',
                    'experience': '',
                    'skills': ''
                }
            }
        }
        
        full_text = ''
        for page in pdf_reader.pages:
            text = page.extract_text()
            full_text += text + '\n\n'
        
        resume_data['content']['full_text'] = full_text
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        parser = PydanticOutputParser(pydantic_object=ResumeContentSchema)
        
        template = """
        You are a resume parser. You are given a resume and you need to extract the following information:
        - Header
        - Education
        - Experience
        - Skills
        
        You need to return a JSON object with the following keys:
        - header
        - education
        - experience
        - skills
        
        output format:
        {{
            "header": "John Doe",
            "education": "University of California, Los Angeles",
            "experience": "Software Engineer at Google",
            "skills": "Python, Java, SQL"
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
        
        resume_data['content']['sections']['header'] = parsed_response.header
        resume_data['content']['sections']['education'] = parsed_response.education
        resume_data['content']['sections']['experience'] = parsed_response.experience
        resume_data['content']['sections']['skills'] = parsed_response.skills
        
        return resume_data

class DocumentProcessor:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, 
                 document_class: str):
        """
        Initialize Document Processor
        
        Args:
            neo4j_uri (str): Neo4j database URI
            neo4j_user (str): Neo4j username
            neo4j_password (str): Neo4j password
            document_class (DocumentClass): Type of document being processed
        """
        self.document_class = DocumentClass.RESUME
        self.neo4j_connection = Neo4jConnection(neo4j_uri, neo4j_user, neo4j_password)
        self.pdf_reader = PDFDocumentReader(self.document_class)

    def create_document_hierarchy(self, filename: str) -> None:
        """
        Create or match document class node and link file node to it
        
        Args:
            filename (str): Name of the PDF file
        """

        query = """
        MERGE (dc:DocumentClass {name: $class_name})
        CREATE (f:File {
            name: $filename,
            created_at: datetime(),
            file_type: 'PDF'
        })
        CREATE (f)-[:BELONGS_TO]->(dc)
        RETURN dc, f
        """
        
        parameters = {
            "class_name": self.document_class.value,
            "filename": filename
        }
        
        try:
            self.neo4j_connection.write_transaction(query, parameters)
            print(f"Successfully created nodes for {filename}")
        except Exception as e:
            print(f"Error creating document hierarchy: {e}")

    def process_document(self, file_path: str) -> None:
        """
        Process document and create graph structure
        
        Args:
            file_path (str): Path to the PDF file
        """
        
        pdf_data = self.pdf_reader.read_pdf(file_path)
        if not pdf_data:
            return

        filename = os.path.basename(file_path)
        
        self.create_document_hierarchy(filename)
        
        if self.document_class == DocumentClass.RESUME:
            self.create_resume_metadata(filename, pdf_data)

    def create_resume_metadata(self, filename: str, pdf_data: dict) -> None:
        """
        Create additional nodes for resume metadata
        
        Args:
            filename (str): Name of the PDF file
            pdf_data (dict): Extracted PDF data
        """
        
        query = """
        MATCH (f:File {name: $filename})
        
        // Create section node and link to file
        CREATE (s:Section {
            name: $section_name,
            content: $content,
            created_at: datetime()
        })
        CREATE (f)-[:HAS_SECTION]->(s)
        """
        
        for section_name, content in pdf_data['content']['sections'].items():
            if content:
                parameters = {
                    "filename": filename,
                    "section_name": section_name,
                    "content": content
                }
                self.neo4j_connection.write_transaction(query, parameters)

if __name__ == "__main__":

    NEO4J_URI = os.getenv('NEO4J_URI_1')
    NEO4J_USERNAME = os.getenv('NEO4J_USERNAME_1')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD_1')
    

    processor = DocumentProcessor(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USERNAME,
        neo4j_password=NEO4J_PASSWORD,
        document_class=DocumentClass.RESUME
    )
    
    processor.process_document("docs/Bob Smith 1.pdf")


