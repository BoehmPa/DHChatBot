import os
import sys
from parameters import *  # Importiere Parameter wie URLs und lokale Verzeichnisse aus einer separaten Datei.
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
    Document
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.web import SimpleWebPageReader
import uvicorn  # Uvicorn wird verwendet, um den FastAPI-Server zu starten.
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel  # Für die Validierung von API-Anfragen und -Antworten
from fastapi.middleware.cors import CORSMiddleware  # Um CORS-Anfragen zuzulassen (Cross-Origin Resource Sharing)

# Setze die LLM- und Embedding-Modelle von Ollama, die für das Chatbot-Modell verwendet werden.
Settings.llm = Ollama(model=OLLAMA_MODEL, request_timeout=300.0, temperature=0.3)
Settings.embed_model = OllamaEmbedding(model_name=OLLAMA_EMBED_MODEL)

# RAGApplication-Klasse, die das System für die Verarbeitung von Anfragen und das Chatten mit dem Bot verwaltet
class RAGApplication:
    def __init__(self):
        self.index = None
        self.chat_engine = None
        self._initialize_index()  # Initialisiert den Index
        self._setup_chat_engine()  # Setzt den Chat-Engine auf

    def _initialize_index(self):
        """Lädt den Index von der Festplatte oder erstellt ihn neu, wenn er nicht existiert."""
        if os.path.exists(PERSIST_DIR):  # Wenn der Index auf der Festplatte existiert
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            self.index = load_index_from_storage(storage_context)  # Lade den Index aus dem Speicher
        else:
            self._ingest_and_create_index()  # Wenn der Index nicht existiert, erstelle ihn neu

    def _ingest_and_create_index(self):
        """Lädt Web- und lokale Daten und indexiert sie."""
        documents = []

        # Lade Webseiten, falls URLs zum Scrapen angegeben sind
        if URLS_TO_SCRAPE:
            print(f"Lade {len(URLS_TO_SCRAPE)} Webseiten")
            try:
                web_reader = SimpleWebPageReader(html_to_text=True)
                web_docs = web_reader.load_data(URLS_TO_SCRAPE)  # Lade Webseiten-Daten
                documents.extend(web_docs)
            except Exception as e:
                print(f"Fehler beim Laden der Webseiten: {e}")

        # Sicherstellen, dass das lokale Datenverzeichnis existiert
        if not os.path.exists(LOCAL_DATA_FOLDER):
            os.makedirs(LOCAL_DATA_FOLDER)

        print(f"Lade lokale Daten aus '{LOCAL_DATA_FOLDER}'")
        local_reader = SimpleDirectoryReader(input_dir=LOCAL_DATA_FOLDER, recursive=True)
        try:
            local_docs = local_reader.load_data()  # Lade lokale Dokumente
            documents.extend(local_docs)
        except Exception as e:
            print(f"Info: {e}")

        if not documents:
            print("Keine Daten gefunden. Erstelle leeren Index.")
            documents = [Document(text="Leeres Initial-Dokument.")]  # Falls keine Dokumente geladen wurden

        # Index erstellen und speichern
        self.index = VectorStoreIndex.from_documents(documents, show_progress=True)
        self.index.storage_context.persist(persist_dir=PERSIST_DIR)  # Speichere den Index

    def _setup_chat_engine(self):
        """Erstellt die Chat-Engine mit Gedächtnis für den Benutzer."""
        memory = ChatMemoryBuffer.from_defaults(token_limit=3900)  # Setze das Gedächtnis des Chats (mit einer Token-Beschränkung)

        # Setze den Chat-Engine mit dem Index und dem Gedächtnis
        self.chat_engine = self.index.as_chat_engine(
            chat_mode="condense_plus_context",  # Modus für das Chatten (mit Kontextverdichtung)
            memory=memory,
            verbose=False,
            system_prompt=("""
                {
                    "system_prompt": {
                        "role": "Hilfreicher Assistent",
                        "main_task": "Du beantwortest Fragen basierend auf den bereitgestellten Dokumenten.",
                        "fallback_behavior": {
                            "condition": "Wenn die Antwort nicht in den Dokumenten gefunden wird",
                            "actions": [
                                "Sage ehrlich, dass die Informationen fehlen.",
                                "Formuliere eine professionelle und präzise E-Mail mit dem Problem an die zuständige Stelle (International Office)."
                            ]
                        }
                    }
                }"""
                )
        )

    def chat(self, user_input: str) -> str:
        """Funktion zur Verarbeitung der Chat-Nachricht des Benutzers."""
        response = self.chat_engine.chat(user_input)  # Verarbeite die Benutzer-Nachricht mit dem Chat-Engine
        return str(response)

    def recommend_countries(self) -> str:
        """Empfiehlt dem Benutzer basierend auf bisherigen Gesprächen bestimmte Länder."""
        prompt = ("""
            {
              "kontext": "Basierend auf unserem gesamten bisherigen Gesprächsverlauf (und NUR darauf):",
              "hauptfrage": "Welche Länder würdest du mir empfehlen zu bereisen oder mich näher damit zu befassen?",
              "anforderung": "Begründe deine Empfehlung kurz anhand der Interessen, die ich im Gespräch geäußert habe."
            }"""
        )
        response = self.chat_engine.chat(prompt)  # Frage den Chat-Engine nach Ländern
        return str(response)


# API-Datenmodelle für die Chat-Anfrage und -Antwort
class ChatRequest(BaseModel):
    """Das JSON, das das Frontend senden muss (Chat-Anfrage)."""
    message: str

class ChatResponse(BaseModel):
    """Das JSON, das die API an das Frontend zurückgibt (Chat-Antwort)."""
    response: str

# Erstelle eine FastAPI-App, die als Schnittstelle für das Backend dient
app_api = FastAPI(
    title="DHBW Heilbronn RAG API",
    description="Eine API für den Chatbot der DHBW Heilbronn.",
    version="1.0.0"
)

# CORS-Middleware hinzufügen, um Cross-Origin-Anfragen zu ermöglichen (WICHTIG für den Zugriff von Frontend-Seiten)
origins = ORIGINS  # Die erlaubten Ursprünge werden in der Datei 'parameters.py' definiert

app_api.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Erlaube Anfragen von den angegebenen Ursprüngen
    allow_credentials=True,
    allow_methods=["*"],  # Erlaube alle HTTP-Methoden
    allow_headers=["*"],  # Erlaube alle Header
)

# Erstelle die Instanz der RAG-Anwendung (um mit dem Chatbot zu interagieren)
app_rag = RAGApplication()

@app_api.get("/")
async def get_root():
    """Ein einfacher Test-Endpunkt, um zu sehen, ob der Server läuft."""
    return {"message": "Willkommen beim Chatbot der DHBW Heilbronn"}

@app_api.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    """
    Haupt-Endpunkt für Chat-Nachrichten.
    Nimmt eine Nachricht entgegen und gibt die Antwort des Bots zurück.
    """
    try:
        bot_response = app_rag.chat(request.message)  # Hole die Antwort des Chatbots
        return ChatResponse(response=bot_response)  # Gebe die Antwort als JSON zurück
    except Exception as e:
        print(f"Fehler bei /chat: {e}")
        raise HTTPException(status_code=500, detail=f"Ein interner Fehler ist aufgetreten: {e}")

# Starte den FastAPI-Server mit Uvicorn, wenn das Skript direkt ausgeführt wird
if __name__ == "__main__":
    print("Starte den API-Server auf http://127.0.0.1:8082")
    print("Die API-Dokumentation findest du unter http://127.0.0.1:8082/docs")
    uvicorn.run(app_api, host="0.0.0.0", port=8082)  # Starte den Server auf allen Interfaces (0.0.0.0)
