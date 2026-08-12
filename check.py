# Перевірка SqliteSaver:
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
 
conn = sqlite3.connect(':memory:')
saver = SqliteSaver(conn)
print(f'SqliteSaver: OK')
 
# Перевірка ChromaDB:
import chromadb
 
client = chromadb.Client()
collection = client.create_collection('test_collection')
collection.add(
    documents=['LangGraph - це фреймворк для агентів', 'ChromaDB - vector database'],
    ids=['doc1', 'doc2'],
)
results = collection.query(query_texts=['агентний фреймворк'], n_results=1)
print(f'ChromaDB query: {results["documents"][0]}')
 
# Перевірка interrupt:
from langgraph.types import interrupt, Command
print(f'interrupt: OK')
