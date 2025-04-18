"""
Outils pour ImproveDoc.

Ce module contient les outils utilisés par ImproveDoc, notamment les outils Wikipedia.
"""

import sys
from typing import List
import wikipedia
from crewai.tools import tool

# Configurer la langue de Wikipedia par défaut
wikipedia.set_lang("fr")

@tool("search_wikipedia")
def search_wikipedia(query: str, results: int = 5) -> List[str]:
    """
    Recherche des pages Wikipedia correspondant à une requête.
    
    Args:
        query: Terme de recherche
        results: Nombre de résultats à retourner
        
    Returns:
        Liste des titres de pages correspondants
    """
    try:
        return wikipedia.search(query, results=results)
    except Exception as e:
        print(f"⚠️ Erreur lors de la recherche Wikipedia: {e}")
        return []

@tool("get_wikipedia_summary")
def get_wikipedia_summary(page_title: str, sentences: int = 5) -> str:
    """
    Récupère le résumé d'une page Wikipedia.
    
    Args:
        page_title: Titre de la page
        sentences: Nombre de phrases à récupérer
        
    Returns:
        Résumé de la page
    """
    try:
        return wikipedia.summary(page_title, sentences=sentences)
    except Exception as e:
        print(f"⚠️ Erreur lors de la récupération du résumé de '{page_title}': {e}")
        return ""

@tool("get_wikipedia_content")
def get_wikipedia_content(page_title: str) -> str:
    """
    Récupère le contenu complet d'une page Wikipedia.
    
    Args:
        page_title: Titre de la page
        
    Returns:
        Contenu complet de la page
    """
    try:
        page = wikipedia.page(page_title)
        return page.content
    except Exception as e:
        print(f"⚠️ Erreur lors de la récupération du contenu de '{page_title}': {e}")
        return ""

@tool("get_wikipedia_url")
def get_wikipedia_url(page_title: str) -> str:
    """
    Récupère l'URL d'une page Wikipedia.
    
    Args:
        page_title: Titre de la page
        
    Returns:
        URL de la page
    """
    try:
        page = wikipedia.page(page_title)
        return page.url
    except Exception as e:
        print(f"⚠️ Erreur lors de la récupération de l'URL de '{page_title}': {e}")
        return ""

# Fonction utilitaire pour changer la langue de Wikipedia
def set_wikipedia_language(language: str):
    """
    Configure la langue utilisée pour les recherches Wikipedia.
    
    Args:
        language: Code de langue pour Wikipedia ("fr", "en", etc.)
    """
    wikipedia.set_lang(language)