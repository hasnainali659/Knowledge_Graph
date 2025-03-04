from neo4j import GraphDatabase

import os

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

# Test query
test_query = """
MERGE (n:Person {name: 'John Doe'})
SET n.age = 30
RETURN n
"""

def main():
    # Create a driver instance
    driver = GraphDatabase.driver(
        NEO4J_URI, 
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

    # Open a session
    with driver.session() as session:
        # Run the test query
        result = session.run(test_query)
        for record in result:
            # Each record contains the RETURN clause from the query
            person_node = record["n"]
            print(f"Created or updated node: {person_node}")

    # Close the driver connection
    driver.close()

if __name__ == "__main__":
    main()
