"""Configuration centrale du projet."""

QDRANT_URL = None              
QDRANT_PATH = "qdrant_data"    
COLLECTION_NAME = "ai_docs"

# Modèle d'embedding multilingue (français + anglais)
EMBED_MODEL = "intfloat/multilingual-e5-small"

# Modèle Ollama 
LLM_MODEL = "qwen2.5:1.5b"

# Découpage en passages (en mots)
CHUNK_SIZE = 150
CHUNK_OVERLAP = 40

# Recherche
TOP_K = 4
MIN_SCORE = 0.79   

DEFAULT_PDF = "data/document.pdf"
