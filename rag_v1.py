import os
import sys
from parameters import *  # Importiert Konfigurationsparameter
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
    Document,
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.web import SimpleWebPageReader

# Konfiguration des LLM- und Embedding-Modells
Settings.llm = Ollama(model=OLLAMA_MODEL, request_timeout=300.0, temperature=0.3)
Settings.embed_model = OllamaEmbedding(model_name=OLLAMA_EMBED_MODEL)

class RAGApplication:
    def __init__(self):
        self.index = None
        self.chat_engine = None
        self._initialize_index()  # Initialisiert den Index
        self._setup_chat_engine()  # Initialisiert die Chat-Engine

    def _initialize_index(self):
        """Lädt den Index, wenn er existiert, oder erstellt ihn neu."""
        if os.path.exists(PERSIST_DIR):
            print(f"Lade existierenden Index aus {PERSIST_DIR}")
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            self.index = load_index_from_storage(storage_context)
        else:
            print("Erstelle neuen Index")
            self._ingest_and_create_index()

    def _ingest_and_create_index(self):
        """Lädt und indexiert lokale und Web-Daten."""
        documents = []

        # Webseiten-Daten laden
        if URLS_TO_SCRAPE:
            print(f"Lade {len(URLS_TO_SCRAPE)} Webseiten")
            try:
                web_reader = SimpleWebPageReader(html_to_text=True)
                web_docs = web_reader.load_data(URLS_TO_SCRAPE)
                documents.extend(web_docs)
            except Exception as e:
                print(f"Fehler beim Laden der Webseiten: {e}")

        # Lokale Dateien laden
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
            documents = [Document(text="Leerer Initial-Dokument.")]

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
                "system_prompt": {
                    "role": "Hilfreicher Assistent",
                    "main_task": "Du beantwortest Fragen basierend auf den bereitgestellten Dokumenten.",
                    "fallback_behavior": {
                        "condition": "Wenn die Antwort nicht in den Dokumenten gefunden wird",
                        "actions": [
                            "Sage ehrlich, dass die Informationen fehlen.",
                            "Formuliere eine professionelle E-Mail an die zuständige Stelle."
                        ]
                    }
                }"""
            )
        )

    def chat(self, user_input: str) -> str:
        """Verarbeitet die Benutzereingabe und gibt eine Antwort zurück."""
        response = self.chat_engine.chat(user_input)
        return str(response)

def start_cli():
    print("Willkommen, ich bin der Chatbot der DHBW Heilbronn")
    app = RAGApplication()
    print("\n=== Chat ist bereit ===")
    print("Tippe '/bye' zum Beenden.\n")

    while True:
        try:
            user_input = input("Du: ")
            if user_input.lower() in ['/bye', 'exit']:
                print("Bye")
                break

            response = app.chat(user_input)
            print(f"\nAssistent:\n{response}\n")

        except KeyboardInterrupt:
            print("\nBeendet durch Nutzer.")
            break
        except Exception as e:
            print(f"Ein Fehler ist aufgetreten: {e}")

if __name__ == "__main__":
    start_cli()
