import os
import sys
from parameters import *
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
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

Settings.llm = Ollama(model=OLLAMA_MODEL, request_timeout=300.0, temperature=0.3)
Settings.embed_model = OllamaEmbedding(model_name=OLLAMA_EMBED_MODEL)



class RAGApplication:
    def __init__(self):
        self.index = None
        self.chat_engine = None
        self._initialize_index()
        self._setup_chat_engine()

    def _initialize_index(self):
        """Lädt den Index von der Festplatte oder erstellt ihn neu."""
        if os.path.exists(PERSIST_DIR):
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            self.index = load_index_from_storage(storage_context)
        else:
            self._ingest_and_create_index()

    def _ingest_and_create_index(self):
        """Lädt Web- und lokale Daten und indexiert sie."""
        documents = []

        # Webseiten laden
        if URLS_TO_SCRAPE:
            print(f"Lade {len(URLS_TO_SCRAPE)} Webseiten")
            try:
                web_reader = SimpleWebPageReader(html_to_text=True)
                web_docs = web_reader.load_data(URLS_TO_SCRAPE)
                documents.extend(web_docs)
            except Exception as e:
                print(f"Fehler beim Laden der Webseiten: {e}")

        # Backup, falls mkdirs.py nicht genutzt wurde
        if not os.path.exists(LOCAL_DATA_FOLDER):
            os.makedirs(LOCAL_DATA_FOLDER)

        print(f"Lade lokale Daten aus '{LOCAL_DATA_FOLDER}'")
        local_reader = SimpleDirectoryReader(input_dir=LOCAL_DATA_FOLDER, recursive=True)
        try:
            local_docs = local_reader.load_data()
            documents.extend(local_docs)
        except Exception as e:
            print(f"Info: {e}")

        if not documents:
            print("Keine Daten gefunden. Erstelle leeren Index.")
            documents = [Document(text="Leeres Initial-Dokument.")]

        # Index erstellen und speichern
        self.index = VectorStoreIndex.from_documents(documents, show_progress=True)
        self.index.storage_context.persist(persist_dir=PERSIST_DIR)

    def _setup_chat_engine(self):
        """Erstellt die Chat-Engine mit Gedächtnis."""
        memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

        self.chat_engine = self.index.as_chat_engine(
            chat_mode="condense_plus_context",
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
        """Normale Chat-Interaktion."""
        response = self.chat_engine.chat(user_input)
        return str(response)

    def recommend_countries(self) -> str:
        prompt = ("""
            {
              "kontext": "Basierend auf unserem gesamten bisherigen Gesprächsverlauf (und NUR darauf):",
              "hauptfrage": "Welche Länder würdest du mir empfehlen zu bereisen oder mich näher damit zu befassen?",
              "anforderung": "Begründe deine Empfehlung kurz anhand der Interessen, die ich im Gespräch geäußert habe."
            }"""
        )
        response = self.chat_engine.chat(prompt)
        return str(response)


# API-Datenmodelle
class ChatRequest(BaseModel):
    """Das JSON, das das Frontend senden muss."""
    message: str

class ChatResponse(BaseModel):
    """Das JSON, das die API an das Frontend zurückgibt."""
    response: str

# FastAPI-App erstellen
app_api = FastAPI(
    title="DHBW Heilbronn RAG API",
    description="Eine API für den Chatbot der DHBW Heilbronn.",
    version="1.0.0"
)

# CORS-Middleware hinzufügen (WICHTIG!)
# Erlaube Anfragen von deinem Frontend, spezifiziert in parameters.py
origins = ORIGINS

app_api.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        bot_response = app_rag.chat(request.message)
        return ChatResponse(response=bot_response)
    except Exception as e:
        print(f"Fehler bei /chat: {e}")
        raise HTTPException(status_code=500, detail=f"Ein interner Fehler ist aufgetreten: {e}")


if __name__ == "__main__":
    print("Starte den API-Server auf http://127.0.0.1:8082")
    print("Die API-Dokumentation findest du unter http://127.0.0.1:8082/docs")
    uvicorn.run(app_api, host="0.0.0.0", port=8082)