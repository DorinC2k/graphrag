// Load nodes
USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
MERGE (n:Entity {id: toInteger(row.id)})
  SET n.name = row.name, n.type = row.type, n.summary = row.summary;

// Load relationships
USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
MATCH (a:Entity {id: toInteger(row.start_id)})
MATCH (b:Entity {id: toInteger(row.end_id)})
CALL apoc.create.relationship(
  a,
  row.type,
  {summary: row.summary, strength: row.strength},
  b
) YIELD rel
RETURN count(rel);
