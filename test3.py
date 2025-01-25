from neo4j import GraphDatabase
import os

NEO4J_URI='neo4j+s://500fcf91.databases.neo4j.io'
NEO4J_USERNAME='neo4j'
NEO4J_PASSWORD='YX5WLEc7qbvZyAf-5lw1TOa9AoGOf5YbeS6_h8RXrd0'
AURA_INSTANCEID='500fcf91'
AURA_INSTANCENAME='Instance01'

# Define the database connection class
class Neo4jConnection:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def run_query(self, query, parameters=None):
        with self._driver.session() as session:
            return session.run(query, parameters)

# Connection parameters
uri = NEO4J_URI  # Change if needed
user = NEO4J_USERNAME  # Replace with your username
password = NEO4J_PASSWORD  # Replace with your password

# Initialize connection
db = Neo4jConnection(uri, user, password)

# Run a sample query
query = "CREATE (n:Person {name: $name, age: $age}) RETURN n"
parameters = {"name": "Alice", "age": 30}

result = db.run_query(query, parameters)

# Process the result
for record in result:
    print(record["n"])

# Close the connection
db.close()
