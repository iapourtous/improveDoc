"""
Outils pour ImproveDoc.

Ce module contient les outils utilisés par ImproveDoc, notamment les outils Wikipedia.
"""

from functools import lru_cache
from typing import List, Optional

import wikipedia
from wikipedia.exceptions import PageError, DisambiguationError
from crewai.tools import tool

# Configurer la langue de Wikipedia par défaut
wikipedia.set_lang("fr")

# Nombre de résultats renvoyés par défaut par l'outil `search_wikipedia`.
# Déclaré avant toute référence afin d'éviter les NameError lors de la
# définition des fonctions décorées.
DEFAULT_RESULTS = 5

# ---------------------------------------------------------------------------
# Fonctions internes avec mise en cache
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1024)
def _cached_search(query: str, results: int = DEFAULT_RESULTS) -> tuple:
    """Appelle ``wikipedia.search`` avec mise en cache LRU."""
    return tuple(wikipedia.search(query, results=results))


@lru_cache(maxsize=256)
def _cached_page(page_title: str):
    """Récupère un objet ``wikipedia.page`` avec mise en cache LRU."""
    return wikipedia.page(page_title, auto_suggest=False, preload=False)


# ---------------------------------------------------------------------------
# Fallback robuste en cas de titre imprécis
# ---------------------------------------------------------------------------


def _get_page_safe(title: str):
    """Retourne une page Wikipédia avec fallback recherche.

    Si le titre exact n'est pas trouvé ou mène à une ambiguïté, tente le premier
    résultat de recherche. Si cela échoue également, lève l'exception d'origine.
    """
    try:
        return _cached_page(title)
    except (PageError, DisambiguationError):
        matches = wikipedia.search(title, results=1)
        if matches:
            return _cached_page(matches[0])
        raise


# ---------------------------------------------------------------------------
# Outillage exposé aux agents CrewAI
# ---------------------------------------------------------------------------


@tool("search_wikipedia")
def search_wikipedia(query: str) -> List[str]:
    """Recherche des pages Wikipedia correspondant à *query*.

    L'implémentation s'appuie sur un cache mémoire LRU afin d'économiser des
    requêtes réseau lorsque plusieurs agents effectuent la même recherche.
    """
    try:
        return list(_cached_search(query, DEFAULT_RESULTS))
    except Exception as e:
        print(f"⚠️ Erreur lors de la recherche Wikipedia: {e}")
        return []


@tool("get_wikipedia_summary")
def get_wikipedia_summary(page_title: str, sentences: int = 5) -> str:
    """Renvoie le résumé *intro* d'une page Wikipedia.

    La fonction exploite le cache interne : si la page a déjà été récupérée par un
    autre outil (contenu ou URL), aucune nouvelle requête réseau n'est déclenchée.
    """
    try:
        return _cached_summary(page_title, sentences)
    except Exception as e:
        print(f"⚠️ Erreur lors de la récupération du résumé de '{page_title}': {e}")
        return ""


@lru_cache(maxsize=1024)
def _cached_summary(page_title: str, sentences: int) -> str:
    """Appelle ``wikipedia.summary`` avec gestion d'erreurs et suggestions.

    En cas d'échec (PageError, DisambiguationError) on tente automatiquement
    d'utiliser le premier résultat de recherche Wikipédia comme fallback. Cela
    permet de récupérer quand le titre fourni n'est pas exact mais suffisamment
    proche.
    """
    try:
        return wikipedia.summary(page_title, sentences=sentences, auto_suggest=False)
    except (PageError, DisambiguationError):
        # Fallback : prendre le premier résultat de recherche
        matches = wikipedia.search(page_title, results=1)
        if matches:
            try:
                return wikipedia.summary(matches[0], sentences=sentences, auto_suggest=False)
            except Exception:
                pass
        raise  # re‑lancer pour gestion par appelant


@tool("get_wikipedia_content")
def get_wikipedia_content(page_title: str) -> str:
    """Renvoie le contenu complet d'une page Wikipedia.

    Utilise un cache LRU afin d'éviter de télécharger plusieurs fois la même page.
    """
    try:
        return _get_page_safe(page_title).content
    except Exception as e:
        print(f"⚠️ Erreur lors de la récupération du contenu de '{page_title}': {e}")
        return ""


@tool("get_wikipedia_url")
def get_wikipedia_url(page_title: str) -> str:
    """Renvoie l'URL canonique vers la page Wikipedia *page_title*."""
    try:
        return _get_page_safe(page_title).url
    except Exception as e:
        print(f"⚠️ Erreur lors de la récupération de l'URL de '{page_title}': {e}")
        return ""


# ---------------------------------------------------------------------------
# Utilitaire public
# ---------------------------------------------------------------------------


def set_wikipedia_language(language: str):
    """Modifie dynamiquement la langue utilisée pour les recherches Wikipedia."""
    wikipedia.set_lang(language)