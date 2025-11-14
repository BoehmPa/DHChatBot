# LLM
OLLAMA_MODEL = "gemma3:12b"

# Modell für Textembeding
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# Ordner für Vektordatenbank
PERSIST_DIR = "./storage_llamaindex"

# Datenquelle der lokalen Daten (z.B. Infobroschüren, Erfahrungsberichte,...) als PDF
LOCAL_DATA_FOLDER = "./datenquelle"

# URLs für die Datenquelle
URLS_TO_SCRAPE = [
    "https://de.wikipedia.org/wiki/Tourismus",

]

# Alle Websites, von denen der Chatbot erreichbar ist
# müssen im Produktivbetrieb ergänzt werden
ORIGINS=[
    "http://localhost:8082",
    "http://localhost",
]