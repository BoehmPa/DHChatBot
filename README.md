# DHChatBot – RAG-basierter Chatbot für die DHBW Heilbronn

Das originale Projekt ist zu finden unter [https://github.com/BoehmPa/DHChatBot](https://github.com/BoehmPa/DHChatBot])

Dieser Chatbot dient als Informations- und Assistenzsystem rund um das Thema **Auslandssemester**.  
Er basiert auf **RAG (Retrieval Augmented Generation)** mit **LlamaIndex** und nutzt **lokale Datenquellen** sowie **Webscraping**.

> **genutzte Ports:** <br>
> Ollama: 11434 <br>
> API: 8082


## Funktionen

- Verarbeitung lokaler PDF-/Text-Dokumente
- Laden und Einbinden externer Webseiten
- Erstellung einer persistenten Vektordatenbank
- Chat mit Gedächtnis (ChatMemoryBuffer)
- CLI-Version (`rag_v1.py`)
- FastAPI-basierte API-Version (`rag_v2.py`)
- CORS-konfigurierbar für Web-Frontends

---

## Projektstruktur

```
.
├── mkdirs.py            # Erstellt lokale Datenordner
├── parameters.py        # Modelle, Ordner, URLs, CORS-Einstellungen
├── rag_v1.py            # CLI-Chatbot
├── rag_v2.py            # FastAPI-Server
├── README.md            # Projektdokumentation
├── requirements.txt     # Abhängigkeiten
├── datenquelle/         # Lokale Dateien für den Index, kann mit mkdirs.py erstellt und manuell befüllt werden
└── storage_llamaindex/  # Speicherort der Vektordatenbank, wird vom Skript erstellt
```

---

## Installation

### 1. Repository klonen
```
git clone https://wi-git.heilbronn.dhbw.de/jan.klitscher.24/chatbot-auslandssemester
cd chatbot-auslandssemester
```
**alternativ**
```
git clone https://github.com/BoehmPa/DHChatBot
cd DHChatBot
```

### 2. Python-Umgebung erzeugen
```
python -m venv .venv
source .venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Abhängigkeiten installieren
```
pip install -r requirements.txt
```

> **Achtung:** Du benötigst eine lokale **Ollama-Installation** und die im Projekt referenzierten Modelle. Wenn du andere modelle verwendest musst du diese in der parameters.py ändern.
> <br> Im Projekt werden diese Modelle verwendet:
>
> - `gemma3:12b`
> - `nomic-embed-text` <br>
> 
---



## Lokale Daten vorbereiten

Falls du lokale Dateien indexieren möchtest:

```
python mkdirs.py
```

Danach PDFs/Textdateien in den Ordner `./datenquelle/` legen.

---

##  Verwendung der CLI-Version

Starte den CLI-Chat:

```
python rag_v1.py
```

Beenden mit:

```
/bye
```

---

## API-Version starten

```
python rag_v2.py
```

Der Server läuft dann unter:

```
http://127.0.0.1:8082
```

API-Dokumentation:

```
http://127.0.0.1:8082/docs
```

---

## Funktionsweise des RAG-Systems

1. **Laden externer Webseiten** (via `SimpleWebPageReader`)
2. **Laden lokaler Dateien**
3. **Erstellen eines Vektorindex** (persistente Speicherung in `./storage_llamaindex`)
4. **ChatEngine mit Gedächtnis erzeugen**
5. **Antworten werden basierend auf Dokumenten generiert**
6. **Fallback-Verhalten**, wenn Informationen fehlen:
   - Ehrliche Aussage, dass Informationen nicht vorhanden sind
   - Generierung einer professionellen E-Mail an das International Office

---

##  Konfiguration (über `parameters.py`)

### KI-Modelle
- `OLLAMA_MODEL`: LLM für den Chat
- `OLLAMA_EMBED_MODEL`: Modell für Embeddings

### Datenquellen
- `LOCAL_DATA_FOLDER`
- `URLS_TO_SCRAPE`

### CORS-Frontend-Freigaben
- `ORIGINS`

---

## Lizenz

Dieses Projekt ist für die Nutzung im Rahmen der DHBW gedacht.

