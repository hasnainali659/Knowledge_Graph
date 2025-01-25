from neo4j import GraphDatabase, TrustSystemCAs  # Add TrustSystemCAs import
from typing import Union

import os

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
        self._connect()
        
    def _connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                trusted_certificates=TrustSystemCAs()  # Changed to TrustSystemCAs()
            )
            # Verify connection is working
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1").single()
            print("Connection to Neo4j DB successful")
        except Exception as e:
            self.driver = None
            raise Exception(f"Failed to connect to Neo4j DB: {e}")
    
    def close(self):
        """Close the Neo4j driver connection"""
        if self.driver is not None:
            self.driver.close()
            self.driver = None
            print("Neo4j connection closed")
    
    def ensure_connection(self):
        """Ensure driver is initialized, attempt reconnection if necessary"""
        if self.driver is None:
            self._connect()
        
    def query(self, query: str, parameters: dict = None) -> Union[list, None]:
        """
        Execute a Cypher query
        
        Args:
            query (str): Cypher query
            parameters (dict): Query parameters (optional)
            
        Returns:
            list: Query results or None if error occurs
        """
        self.ensure_connection()
        session = None
        response = None
        
        try:
            session = self.driver.session(database=self.database)
            result = session.run(query, parameters or {})
            response = [record.data() for record in result]
        except Exception as e:
            print(f"Query failed: {e}")
            raise
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
        self.ensure_connection()
        session = None
        
        try:
            session = self.driver.session(database=self.database)
            session.execute_write(
                lambda tx: tx.run(query, parameters or {})
            )
        except Exception as e:
            print(f"Write transaction failed: {e}")
            raise
        finally:
            if session is not None:
                session.close()

# Usage example with proper error handling
try:
    # Connection details
    NEO4J_URI = os.getenv('NEO4J_URI_1')
    NEO4J_USERNAME = os.getenv('NEO4J_USERNAME_1')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD_1')

    # Create connection
    conn = Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    
    # Test query
    test_query = """
    MERGE (n:Person {name: 'John Doe'})
    SET n.age = 30
    RETURN n
    """
    
    result = conn.query(test_query)
    print(f"Query result: {result}")
    
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if 'conn' in locals():
        conn.close()