import os
from parameters import*
"""
Erstellt den Ordner für die Datenquelle. 
Der Ordner für die Vektordatenbank wird vom rag_v1/v2 Skript erstellt.
"""
if not os.path.exists(LOCAL_DATA_FOLDER):
    os.makedirs(LOCAL_DATA_FOLDER)