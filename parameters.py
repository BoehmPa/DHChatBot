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
    "https://gostralia-gomerica.de/universitaet/australien/griffith-university-gold-coast",
    "https://gostralia-gomerica.de/universitaet/australien/queensland-university-of-technology-brisbane",
    "https://gostralia-gomerica.de/universitaet/australien/southern-cross-university-gold-coast",
    "https://gostralia-gomerica.de/universitaet/australien/james-cook-university-townsville",
    "https://gostralia-gomerica.de/universitaet/australien/university-of-the-sunshine-coast-maroochydore",
    "https://gostralia-gomerica.de/universitaet/australien/university-of-technology-sydney",
    "https://gostralia-gomerica.de/universitaet/australien/unsw-sydney",
    "https://gostralia-gomerica.de/universitaet/neuseeland/university-of-auckland",
    "https://gostralia-gomerica.de/universitaet/neuseeland/auckland-university-of-technology",
    "https://gostralia-gomerica.de/universitaet/neuseeland/university-of-waikato-hamilton",
    "https://gostralia-gomerica.de/universitaet/neuseeland/massey-university",
    "https://gostralia-gomerica.de/universitaet/singapur/james-cook-university-singapur",
    "https://gostralia-gomerica.de/universitaet/malaysia/monash-university-malaysia",
    "https://gostralia-gomerica.de/universitaet/malaysia/swinburne-university-of-technology-sarawak",
    "https://gostralia-gomerica.de/universitaet/vietnam/rmit-university-vietnam",
    "https://gostralia-gomerica.de/universitaet/usa/california-state-university-northridge",
    "https://gostralia-gomerica.de/universitaet/usa/hawaii-pacific-university",
    "https://gostralia-gomerica.de/universitaet/usa/san-diego-state-university",
    "https://gostralia-gomerica.de/universitaet/usa/university-of-california-berkeley-extension",
    "https://www.savonia.fi/en/study-with-us/for-students/new-exchange-students/",
    "https://opinto-opas.peppi.savonia.fi/offering/27/64899?lang=en",
    "https://www.erasmusplus.de/erasmus/hochschulbildung",
    "https://www.auswaertiges-amt.de/de/reiseundsicherheit/10-2-8reisewarnungen ",
]

# Alle Websites, von denen der Chatbot erreichbar ist
# müssen im Produktivbetrieb ergänzt werden
ORIGINS=[
    "http://localhost:8082",
    "http://localhost",
]