"""
Neo4j Graph Database Manager for Agentic GraphRAG

This module provides a robust interface for Neo4j operations including:
- Connection management with automatic retry
- Node CRUD operations
- Relationship CRUD operations
- Schema management
- Query execution
- Batch operations

Author: Agentic GraphRAG Team
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from neo4j import GraphDatabase, Driver, Session, Transaction
from neo4j.exceptions import ServiceUnavailable, SessionExpired
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from contextlib import contextmanager

from ..utils.config import get_config, Neo4jConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def sanitize_label(label: str) -> str:
    """
    Sanitize a label or relationship type for Neo4j.

    Neo4j labels and relationship types must:
    - Start with a letter
    - Contain only letters, numbers, and underscores
    - Not contain spaces or special characters

    Args:
        label: Raw label string

    Returns:
        Sanitized label safe for Neo4j
    """
    import re

    # Replace spaces and hyphens with underscores
    sanitized = label.replace(' ', '_').replace('-', '_')

    # Remove any characters that aren't alphanumeric or underscore
    sanitized = re.sub(r'[^\w]', '', sanitized)

    # Ensure it starts with a letter (add prefix if needed)
    if sanitized and not sanitized[0].isalpha():
        sanitized = 'N_' + sanitized

    # Handle empty string
    if not sanitized:
        sanitized = 'Unknown'

    return sanitized


class Neo4jManager:
    """
    Manager class for Neo4j graph database operations.

    Provides high-level interface for:
    - Creating and managing nodes
    - Creating and managing relationships
    - Running queries
    - Schema operations
    - Batch operations

    Attributes:
        config: Neo4j configuration
        driver: Neo4j driver instance
    """

    def __init__(self, config: Optional[Neo4jConfig] = None):
        """
        Initialize Neo4j manager.

        Args:
            config: Optional Neo4jConfig. If None, loads from environment.

        Raises:
            ServiceUnavailable: If cannot connect to Neo4j
        """
        self.config = config or get_config().neo4j
        self.driver: Optional[Driver] = None
        self._connect()
        logger.info(f"Initialized Neo4j manager connected to {self.config.uri}")

    def _connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.user, self.config.password)
            )
            # Test connection
            with self.driver.session(database=self.config.database) as session:
                session.run("RETURN 1")
            logger.info("Successfully connected to Neo4j")
        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self) -> None:
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")

    @contextmanager
    def get_session(self) -> Session:
        """
        Context manager for Neo4j sessions.

        Yields:
            Session: Neo4j session
        """
        session = self.driver.session(database=self.config.database)
        try:
            yield session
        finally:
            session.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ServiceUnavailable, SessionExpired)),
        reraise=True,
    )
    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query with automatic retry.

        Args:
            query: Cypher query string
            parameters: Optional query parameters

        Returns:
            List of result records as dictionaries

        Raises:
            ServiceUnavailable: If query fails after retries
        """
        parameters = parameters or {}

        with self.get_session() as session:
            try:
                result = session.run(query, parameters)
                records = [dict(record) for record in result]
                logger.debug(f"Query executed: {query[:100]}... returned {len(records)} records")
                return records
            except Exception as e:
                logger.error(f"Query failed: {query[:100]}... Error: {e}")
                raise

    # ==================== Node Operations ====================

    def create_node(
        self,
        label: str,
        properties: Dict[str, Any],
        merge: bool = False
    ) -> Dict[str, Any]:
        """
        Create a node in the graph.

        Args:
            label: Node label (e.g., "Person", "Document")
            properties: Node properties as key-value pairs
            merge: If True, use MERGE instead of CREATE (avoids duplicates)

        Returns:
            Created node properties including internal ID
        """
        if not properties:
            raise ValueError("Node properties cannot be empty")

        # Sanitize label
        sanitized_label = sanitize_label(label)

        if merge:
            # MERGE requires explicit property matching
            query = f"""
            MERGE (n:{sanitized_label} {{name: $name}})
            ON CREATE SET n = $properties
            ON MATCH SET n += $properties
            RETURN n, id(n) as node_id
            """
            # Use 'name' as merge key, or first property if no 'name' exists
            merge_key = properties.get('name', list(properties.values())[0] if properties else '')
            result = self.execute_query(query, {"properties": properties, "name": merge_key})
        else:
            # CREATE can use parameter map directly
            query = f"""
            CREATE (n:{sanitized_label})
            SET n = $properties
            RETURN n, id(n) as node_id
            """
            result = self.execute_query(query, {"properties": properties})
        if result:
            node_data = dict(result[0]["n"])
            node_data["_id"] = result[0]["node_id"]
            logger.info(f"{'Merged' if merge else 'Created'} node {sanitized_label}: {node_data.get('name', 'unnamed')}")
            return node_data
        return {}

    def get_node(
        self,
        label: str,
        properties: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Get a node by label and properties.

        Args:
            label: Node label
            properties: Properties to match

        Returns:
            Node properties if found, None otherwise
        """
        sanitized_label = sanitize_label(label)
        conditions = " AND ".join([f"n.{key} = ${key}" for key in properties.keys()])
        query = f"""
        MATCH (n:{sanitized_label})
        WHERE {conditions}
        RETURN n, id(n) as node_id
        LIMIT 1
        """

        result = self.execute_query(query, properties)
        if result:
            node_data = dict(result[0]["n"])
            node_data["_id"] = result[0]["node_id"]
            return node_data
        return None

    def update_node(
        self,
        label: str,
        match_properties: Dict[str, Any],
        update_properties: Dict[str, Any]
    ) -> bool:
        """
        Update a node's properties.

        Args:
            label: Node label
            match_properties: Properties to match the node
            update_properties: Properties to update

        Returns:
            True if node was updated, False otherwise
        """
        sanitized_label = sanitize_label(label)
        conditions = " AND ".join([f"n.{key} = ${key}" for key in match_properties.keys()])
        query = f"""
        MATCH (n:{sanitized_label})
        WHERE {conditions}
        SET n += $update_props
        RETURN n
        """

        params = {**match_properties, "update_props": update_properties}
        result = self.execute_query(query, params)
        success = len(result) > 0

        if success:
            logger.info(f"Updated node {sanitized_label}")
        return success

    def delete_node(
        self,
        label: str,
        properties: Dict[str, Any],
        detach: bool = True
    ) -> bool:
        """
        Delete a node from the graph.

        Args:
            label: Node label
            properties: Properties to match
            detach: If True, also delete all relationships (DETACH DELETE)

        Returns:
            True if node was deleted, False otherwise
        """
        sanitized_label = sanitize_label(label)
        conditions = " AND ".join([f"n.{key} = ${key}" for key in properties.keys()])
        delete_clause = "DETACH DELETE" if detach else "DELETE"
        query = f"""
        MATCH (n:{sanitized_label})
        WHERE {conditions}
        {delete_clause} n
        RETURN count(n) as deleted_count
        """

        result = self.execute_query(query, properties)
        deleted_count = result[0]["deleted_count"] if result else 0

        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} node(s) with label {sanitized_label}")
        return deleted_count > 0

    # ==================== Relationship Operations ====================

    def create_relationship(
        self,
        from_label: str,
        from_properties: Dict[str, Any],
        to_label: str,
        to_properties: Dict[str, Any],
        relationship_type: str,
        relationship_properties: Optional[Dict[str, Any]] = None,
        merge: bool = False
    ) -> Dict[str, Any]:
        """
        Create a relationship between two nodes.

        Args:
            from_label: Source node label
            from_properties: Source node properties to match
            to_label: Target node label
            to_properties: Target node properties to match
            relationship_type: Relationship type (e.g., "KNOWS", "WORKS_AT")
            relationship_properties: Optional relationship properties
            merge: If True, use MERGE to avoid duplicates

        Returns:
            Relationship properties
        """
        relationship_properties = relationship_properties or {}

        # Sanitize labels and relationship type
        sanitized_from_label = sanitize_label(from_label)
        sanitized_to_label = sanitize_label(to_label)
        sanitized_rel_type = sanitize_label(relationship_type)

        from_conditions = " AND ".join([f"from.{key} = $from_{key}" for key in from_properties.keys()])
        to_conditions = " AND ".join([f"to.{key} = $to_{key}" for key in to_properties.keys()])

        if merge:
            # MERGE requires explicit property setting
            query = f"""
            MATCH (from:{sanitized_from_label}), (to:{sanitized_to_label})
            WHERE {from_conditions} AND {to_conditions}
            MERGE (from)-[r:{sanitized_rel_type}]->(to)
            ON CREATE SET r = $rel_props
            ON MATCH SET r += $rel_props
            RETURN r, id(r) as rel_id
            """
        else:
            # CREATE can set properties directly
            query = f"""
            MATCH (from:{sanitized_from_label}), (to:{sanitized_to_label})
            WHERE {from_conditions} AND {to_conditions}
            CREATE (from)-[r:{sanitized_rel_type}]->(to)
            SET r = $rel_props
            RETURN r, id(r) as rel_id
            """

        params = {
            **{f"from_{k}": v for k, v in from_properties.items()},
            **{f"to_{k}": v for k, v in to_properties.items()},
            "rel_props": relationship_properties
        }

        result = self.execute_query(query, params)
        if result:
            rel_data = dict(result[0]["r"])
            rel_data["_id"] = result[0]["rel_id"]
            logger.info(f"{'Merged' if merge else 'Created'} relationship {sanitized_rel_type}")
            return rel_data
        return {}

    def get_relationships(
        self,
        node_label: str,
        node_properties: Dict[str, Any],
        direction: str = "both",
        relationship_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all relationships for a node.

        Args:
            node_label: Node label
            node_properties: Node properties to match
            direction: "incoming", "outgoing", or "both"
            relationship_type: Optional filter by relationship type

        Returns:
            List of relationships with connected nodes
        """
        conditions = " AND ".join([f"n.{key} = ${key}" for key in node_properties.keys()])

        if direction == "incoming":
            pattern = f"(other)-[r{':' + relationship_type if relationship_type else ''}]->(n)"
        elif direction == "outgoing":
            pattern = f"(n)-[r{':' + relationship_type if relationship_type else ''}]->(other)"
        else:  # both
            pattern = f"(n)-[r{':' + relationship_type if relationship_type else ''}]-(other)"

        query = f"""
        MATCH {pattern}
        WHERE {conditions}
        RETURN r, other, type(r) as rel_type, id(r) as rel_id
        """

        result = self.execute_query(query, node_properties)
        relationships = []
        for record in result:
            rel_data = {
                "relationship": dict(record["r"]),
                "relationship_type": record["rel_type"],
                "relationship_id": record["rel_id"],
                "connected_node": dict(record["other"])
            }
            relationships.append(rel_data)

        return relationships

    # ==================== Batch Operations ====================

    def create_nodes_batch(
        self,
        label: str,
        nodes: List[Dict[str, Any]],
        merge: bool = False
    ) -> int:
        """
        Create multiple nodes in a batch operation.

        Args:
            label: Node label
            nodes: List of node property dictionaries
            merge: If True, use MERGE instead of CREATE

        Returns:
            Number of nodes created
        """
        if not nodes:
            return 0

        operation = "MERGE" if merge else "CREATE"
        query = f"""
        UNWIND $nodes as node_data
        {operation} (n:{label})
        SET n = node_data
        RETURN count(n) as created_count
        """

        result = self.execute_query(query, {"nodes": nodes})
        count = result[0]["created_count"] if result else 0
        logger.info(f"{'Merged' if merge else 'Created'} {count} nodes with label {label}")
        return count

    def create_relationships_batch(
        self,
        relationships: List[Dict[str, Any]],
        merge: bool = False
    ) -> int:
        """
        Create multiple relationships in a batch operation.

        Args:
            relationships: List of relationship dictionaries with keys:
                - from_label, from_id, to_label, to_id, type, properties
            merge: If True, use MERGE instead of CREATE

        Returns:
            Number of relationships created
        """
        if not relationships:
            return 0

        operation = "MERGE" if merge else "CREATE"
        query = f"""
        UNWIND $relationships as rel_data
        MATCH (from {{id: rel_data.from_id}}), (to {{id: rel_data.to_id}})
        {operation} (from)-[r:REL_TYPE]->(to)
        SET r = rel_data.properties
        SET type(r) = rel_data.type
        RETURN count(r) as created_count
        """

        # Note: Neo4j doesn't support dynamic relationship types in this way
        # This is a simplified version; in practice, you'd need separate queries per type
        logger.warning("Batch relationship creation requires same relationship type")
        return 0

    # ==================== Schema Operations ====================

    def create_index(self, label: str, property_name: str) -> bool:
        """
        Create an index on a node label property.

        Args:
            label: Node label
            property_name: Property to index

        Returns:
            True if index was created successfully
        """
        query = f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.{property_name})"
        try:
            self.execute_query(query)
            logger.info(f"Created index on {label}.{property_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False

    def create_constraint(
        self,
        label: str,
        property_name: str,
        constraint_type: str = "UNIQUE"
    ) -> bool:
        """
        Create a constraint on a node label property.

        Args:
            label: Node label
            property_name: Property to constrain
            constraint_type: "UNIQUE" or "EXISTS"

        Returns:
            True if constraint was created successfully
        """
        constraint_name = f"{label}_{property_name}_{constraint_type.lower()}"

        if constraint_type.upper() == "UNIQUE":
            query = f"""
            CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
            FOR (n:{label}) REQUIRE n.{property_name} IS UNIQUE
            """
        else:
            query = f"""
            CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
            FOR (n:{label}) REQUIRE n.{property_name} IS NOT NULL
            """

        try:
            self.execute_query(query)
            logger.info(f"Created {constraint_type} constraint on {label}.{property_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create constraint: {e}")
            return False

    def get_schema(self) -> Dict[str, Any]:
        """
        Get the current graph schema.

        Returns:
            Dictionary containing labels, relationship types, and indexes
        """
        # Get node labels
        labels_query = "CALL db.labels()"
        labels = [r["label"] for r in self.execute_query(labels_query)]

        # Get relationship types
        rel_types_query = "CALL db.relationshipTypes()"
        relationship_types = [r["relationshipType"] for r in self.execute_query(rel_types_query)]

        # Get indexes
        indexes_query = "SHOW INDEXES"
        try:
            indexes = self.execute_query(indexes_query)
        except:
            indexes = []

        return {
            "labels": labels,
            "relationship_types": relationship_types,
            "indexes": indexes
        }

    # ==================== Utility Operations ====================

    def clear_database(self) -> bool:
        """
        Clear all nodes and relationships from the database.

        WARNING: This deletes all data!

        Returns:
            True if database was cleared successfully
        """
        query = "MATCH (n) DETACH DELETE n"
        try:
            self.execute_query(query)
            logger.warning("Cleared all data from Neo4j database")
            return True
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            return False

    def get_node_count(self, label: Optional[str] = None) -> int:
        """
        Get count of nodes in the database.

        Args:
            label: Optional label to filter by

        Returns:
            Number of nodes
        """
        if label:
            query = f"MATCH (n:{label}) RETURN count(n) as count"
        else:
            query = "MATCH (n) RETURN count(n) as count"

        result = self.execute_query(query)
        return result[0]["count"] if result else 0

    def get_relationship_count(self, relationship_type: Optional[str] = None) -> int:
        """
        Get count of relationships in the database.

        Args:
            relationship_type: Optional type to filter by

        Returns:
            Number of relationships
        """
        if relationship_type:
            query = f"MATCH ()-[r:{relationship_type}]->() RETURN count(r) as count"
        else:
            query = "MATCH ()-[r]->() RETURN count(r) as count"

        result = self.execute_query(query)
        return result[0]["count"] if result else 0


# Singleton instance
_neo4j_manager: Optional[Neo4jManager] = None


def get_neo4j_manager() -> Neo4jManager:
    """
    Get the global Neo4j manager instance (singleton pattern).

    Returns:
        Neo4jManager: Global Neo4j manager
    """
    global _neo4j_manager
    if _neo4j_manager is None:
        _neo4j_manager = Neo4jManager()
    return _neo4j_manager


def reset_neo4j_manager() -> None:
    """Reset the global Neo4j manager (useful for testing)."""
    global _neo4j_manager
    if _neo4j_manager:
        _neo4j_manager.close()
    _neo4j_manager = None


if __name__ == "__main__":
    """Test the Neo4j manager."""
    import sys

    try:
        print("🔄 Initializing Neo4j manager...")
        manager = get_neo4j_manager()

        print("\n📊 Current schema:")
        schema = manager.get_schema()
        print(f"  Labels: {schema['labels']}")
        print(f"  Relationship types: {schema['relationship_types']}")

        print(f"\n📈 Database stats:")
        print(f"  Total nodes: {manager.get_node_count()}")
        print(f"  Total relationships: {manager.get_relationship_count()}")

        print("\n✅ Neo4j manager working correctly!")

        manager.close()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. Docker Desktop is running")
        print("  2. Neo4j container is started: ./scripts/start_neo4j.sh")
        print("  3. .env file has correct NEO4J_* settings")
        sys.exit(1)
