"""
build_relationships.py
Creates RELATED_TO relationships in Neo4j from the DEDuCT_KG edge TSV files.
Each edge file encodes: SourceLabel.relation.TargetLabel.edges.tsv
Nodes are matched by their 'doc_id' property (stored as "node::<id>").
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from neo4j import GraphDatabase

load_dotenv(r"C:\Code\Mycode\DEDuCTv3.0\.env")

URI      = os.getenv("NEO4J_URI")
USER     = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")

EDGE_DIR = Path(r"C:\Code\Mycode\DEDuCTv3.0\Supporting_Data\DEDuCT_KG\edge_tables")
BATCH_SIZE = 500   # rows per transaction


def parse_edge_filename(path: Path):
    pieces = path.name.split(".")
    # Format: SourceType.relation_part.TargetType.edges.tsv  (>=5 parts)
    if len(pieces) < 5:
        raise ValueError(f"Unexpected filename: {path.name}")
    source_type = pieces[0]
    target_type = pieces[-3]
    relation    = ".".join(pieces[1:-3])
    return source_type, relation, target_type


def create_relationships(session, rows, source_type, target_type, relation):
    """
    Batch-upsert RELATED_TO edges.
    Nodes are matched by their raw 'id' column stored as doc_id = 'node::<id>'.
    """
    # Add the labels to the MATCH clause so Neo4j uses the index
    cypher = (
        f"UNWIND $rows AS row "
        f"MATCH (src:{source_type} {{doc_id: 'node::' + row.source}}) "
        f"MATCH (tgt:{target_type} {{doc_id: 'node::' + row.target}}) "
        f"MERGE (src)-[r:RELATED_TO {{relation: row.relation}}]->(tgt) "
        f"SET r.source_type = row.source_type, r.target_type = row.target_type"
    )
    session.run(cypher, rows=rows)


def create_indexes(driver):
    print("\nCreating Search Indexes...")
    with driver.session() as s:
        labels = ["Chemical", "Gene", "Disease", "AOP", "Phenotype", "KeyEvent", "DEDuCT_Endpoint", "Generic"]
        for label in labels:
            # 1. Full-Text Index
            ft_index = f"{label.lower()}_text_index"
            exists_ft = s.run(f"SHOW INDEXES YIELD name, type WHERE name = '{ft_index}' RETURN count(*) AS count").single()["count"]
            if not exists_ft:
                try:
                    s.run(f"CREATE FULLTEXT INDEX {ft_index} FOR (n:{label}) ON EACH [n.name, n.text]")
                    print(f"  [+] Created FULLTEXT index: {ft_index}")
                except Exception as e:
                    print(f"  [!] Failed {ft_index}: {e}")
            else:
                print(f"  [-] FULLTEXT index {ft_index} already exists")

            # 2. Vector Index (ensure missing ones like Generic are created)
            vec_index = f"{label.lower()}_embedding_index"
            exists_vec = s.run(f"SHOW INDEXES YIELD name, type WHERE name = '{vec_index}' RETURN count(*) AS count").single()["count"]
            if not exists_vec:
                try:
                    s.run(
                        f"CREATE VECTOR INDEX {vec_index} FOR (n:{label}) ON (n.embedding) "
                        f"OPTIONS {{indexConfig: {{`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}}}"
                    )
                    print(f"  [+] Created VECTOR index: {vec_index}")
                except Exception as e:
                    print(f"  [!] Failed {vec_index}: {e}")
            else:
                print(f"  [-] VECTOR index {vec_index} already exists")

def main():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    edge_files = sorted(EDGE_DIR.glob("*.edges.tsv"))
    print(f"Found {len(edge_files)} edge files.")

    total_created = 0
    for ef in edge_files:
        source_type, relation, target_type = parse_edge_filename(ef)
        df = pd.read_csv(ef, sep="\t", dtype=str, keep_default_na=False)
        if "source" not in df.columns or "target" not in df.columns:
            print(f"  SKIP {ef.name}: missing source/target columns")
            continue

        # Drop rows missing source or target
        df = df[df["source"].str.strip() != ""]
        df = df[df["target"].str.strip() != ""]
        total_rows = len(df)
        print(f"Processing {ef.name}  ({total_rows} edges, relation={relation}) ...", end=" ", flush=True)

        batch_data = []
        created = 0
        with driver.session() as session:
            for _, row in df.iterrows():
                batch_data.append({
                    "source":      row["source"].strip(),
                    "target":      row["target"].strip(),
                    "relation":    relation,
                    "source_type": source_type,
                    "target_type": target_type,
                })
                if len(batch_data) >= BATCH_SIZE:
                    create_relationships(session, batch_data, source_type, target_type, relation)
                    created += len(batch_data)
                    batch_data = []
            if batch_data:
                create_relationships(session, batch_data, source_type, target_type, relation)
                created += len(batch_data)

        print(f"done ({created} processed)")
        total_created += created

    print(f"\nFinished. Total edges processed: {total_created}")
    
    # Run index creation automatically after data ingestion
    create_indexes(driver)
    
    driver.close()


if __name__ == "__main__":
    main()
