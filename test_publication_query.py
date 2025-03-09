import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Connect to Neo4j
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

# Verify we have credentials
print(f"URI: {uri}")
print(f"Username: {username}")
print(f"Password: {'*' * len(password) if password else 'None'}")

# Create a driver
driver = GraphDatabase.driver(uri, auth=(username, password))

def get_publications():
    with driver.session() as session:
        # First, check if the publication node exists
        check_query = """
        MATCH (p:PUBLICATIONS {name: 'Hasnain Ali Poonja_publications'})
        RETURN p
        """
        check_result = session.run(check_query)
        if not check_result.peek():
            print("No publications node found with the name 'Hasnain Ali Poonja_publications'")
            # Try to find any publications nodes
            all_pubs_query = """
            MATCH (p:PUBLICATIONS)
            RETURN p.name
            """
            all_pubs = session.run(all_pubs_query)
            print("Available publication nodes:")
            for record in all_pubs:
                print(f" - {record['p.name']}")
            return
        
        # Try the corrected query
        correct_query = """
        MATCH (target:FILE {name: 'Hasnain Ali Resume.pdf'})
        CALL apoc.path.subgraphAll(target, {maxLevel: 6}) YIELD nodes, relationships
        WITH nodes, relationships
        UNWIND nodes as node
        WITH node
        WHERE node:PUBLICATIONS AND node.name = 'Hasnain Ali Poonja_publications'
        RETURN node.name
        """
        
        # Execute correct query
        print("\nTrying the corrected query...")
        result = session.run(correct_query)
        publications = [record["node.name"] for record in result]
        print(f"Publications found: {publications}")
        
        # If we found the publication node, let's try to get its connected items
        if publications:
            items_query = """
            MATCH (p:PUBLICATIONS {name: 'Hasnain Ali Poonja_publications'})-[:HAS_VALUE]->(i:ITEM)
            RETURN i.name
            """
            items_result = session.run(items_query)
            print("\nPublication items:")
            for record in items_result:
                print(f" - {record['i.name']}")

try:
    get_publications()
    print("\nQuery executed successfully")
except Exception as e:
    print(f"\nError: {e}")
finally:
    driver.close() 