# Mage Orchestration Notes

This directory is mounted into the `mage` service defined in `docker-compose.yml`. It contains user code blocks and pipeline definitions that Mage can execute. Use the Mage UI (<http://localhost:6789>) to create a new project pointing at `/home/src` and link the blocks below.

Suggested pipeline layout:

1. **load_gdelt_block.py** – Downloads the latest GDELT GKG file or reads the cached sample dataset from `data/sample_articles.jsonl`.
2. **ner_block.py** – Invokes the Hugging Face NER pipeline defined in `src/nlp/ner.py`.
3. **entity_resolution_block.py** – Calls the resolver (`src/nlp/entity_resolution.py`).
4. **neo4j_loader_block.py** – Persists the results using `src/graph/neo4j_loader.py`.
5. **export_block.py** – Runs `src/graph/export.py` to create the PyG snapshot.

Blocks can import project code with `from src...` because `/home/src` is part of `PYTHONPATH` inside the Mage container.
