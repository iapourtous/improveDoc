"""
Configuration pour ImproveDoc.

Ce module gère la configuration et les variables d'environnement.
"""

import os
import sys
from typing import Optional
from dotenv import load_dotenv

# Désactiver la télémétrie
os.environ["CREWAI_TELEMETRY"] = "False"
os.environ["TELEMETRY_ENABLED"] = "False"
os.environ["OPENTELEMETRY_ENABLED"] = "False"

# Supprimer les modules de télémétrie s'ils sont chargés
for mod in list(sys.modules.keys()):
    if 'telemetry' in mod or 'opentelemetry' in mod:
        if mod in sys.modules:
            del sys.modules[mod]

# Charger les variables d'environnement
load_dotenv()

class Config:
    """
    Gestion de la configuration pour ImproveDoc.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialise la configuration.
        
        Args:
            api_key: Clé API pour le modèle LLM (optionnel, sinon utilise la variable d'environnement)
            model: Nom du modèle à utiliser (optionnel, sinon utilise la variable d'environnement)
        """
        # Configuration LLM
        self.api_key = api_key or os.getenv('CREW_API_KEY')
        self.base_url = os.getenv('CREW_BASE_URL', 'https://openrouter.ai/api/v1')
        self.model = model or os.getenv('CREW_MODEL', 'openrouter/openai/gpt-4.1-mini')
        
        # Paramètres LLM
        self.temperature = float(os.getenv('CREW_TEMPERATURE', '0.7'))
        self.max_tokens = int(os.getenv('CREW_MAX_TOKENS', '4096'))
        
        # Configuration Wikipedia
        self.wikipedia_language = os.getenv('WIKI_LANGUAGE', 'fr')
    
    def validate(self) -> bool:
        """
        Valide que la configuration est correcte.
        
        Returns:
            True si la configuration est valide, False sinon
        """
        if not self.api_key:
            print("⚠️ Aucune clé API n'a été configurée.")
            print("Définissez la variable d'environnement CREW_API_KEY, ou utilisez l'option --api-key")
            return False
        return True
    
    def is_debug_enabled(self) -> bool:
        """
        Vérifie si le mode debug est activé.
        
        Returns:
            True si le mode debug est activé, False sinon
        """
        return os.getenv('DEBUG', 'False').lower() in ('true', '1', 't')