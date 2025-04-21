#!/usr/bin/env python3
"""
CreateDoc - Outil de création de documents Markdown structurés

Cet outil permet de créer un document Markdown structuré à partir d'un titre,
d'un sujet et d'instructions spécifiques, puis de l'améliorer automatiquement
avec ImproveDoc.
"""

import sys
import os
import argparse
from pathlib import Path
import logging
import warnings
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# Ajouter le répertoire courant au chemin pour permettre les imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Désactiver les warnings
warnings.filterwarnings("ignore", category=Warning)

try:
    # Importer les modules nécessaires
    from lib.enhancer import ContentEnhancer
    from lib.creator import ContentCreator
    from lib.agents import AgentFactory
    from crewai import LLM
    
    # Vérifier que les modules nécessaires sont installés
    import crewai
    import wikipedia
except ImportError as e:
    print(f"⚠️ Erreur d'importation: {e}")
    print("⚠️ Certaines dépendances requises ne sont pas installées.")
    print("Installez-les avec : pip install -r requirements.txt")
    print("Assurez-vous d'activer l'environnement virtuel : source venv/bin/activate")
    sys.exit(1)

class DocumentCreator:
    """
    Classe responsable de la création et l'amélioration de documents Markdown structurés
    basée sur les principes SOLID.
    """
    
    def __init__(self, api_key=None, model=None):
        """
        Initialise le créateur de documents.
        
        Args:
            api_key: Clé API pour le modèle LLM (optionnel)
            model: Nom du modèle LLM à utiliser (optionnel)
        """
        self.api_key = api_key
        self.model = model
        
        # Ne pas créer de LLM ici, les services en créeront eux-mêmes
        
        # Ne pas initialiser de factory d'agents ici
        
        # Initialiser les services directement avec les clés API
        # Exactement comme dans improvedoc.py
        try:
            self.content_enhancer = ContentEnhancer(api_key=api_key, model=model)
            
            # Créer le service de création de contenu avec une factory d'agents
            # Réutiliser la factory du ContentEnhancer pour être cohérent
            self.content_creator = ContentCreator(
                agent_factory=self.content_enhancer.agent_factory,
                api_key=api_key,
                model=model
            )
        except ValueError as e:
            print(f"⚠️ Erreur de configuration: {e}")
            raise
    
    def create_document(self, title, subject="", instructions=""):
        """
        Crée un document Markdown structuré basé sur les entrées utilisateur
        en utilisant les agents spécialisés pour générer un contenu initial de qualité.
        
        Args:
            title: Titre du document
            subject: Sujet principal du document (facultatif)
            instructions: Instructions spécifiques pour le contenu (facultatif)
            
        Returns:
            str: Contenu Markdown généré
        """
        print("📝 Génération du contenu initial avec des agents spécialisés...")
        return self.content_creator.create_content(title, subject, instructions)
    
    def improve_document(self, content):
        """
        Améliore le contenu du document en utilisant ContentEnhancer.
        
        Args:
            content: Contenu Markdown à améliorer
            
        Returns:
            str: Contenu Markdown amélioré
        """
        try:
            print("🔍 Amélioration du contenu avec des informations de Wikipedia...")
            improved_content = self.content_enhancer.enhance_content(content)
            return improved_content
        except Exception as e:
            print(f"⚠️ Erreur lors de l'amélioration du contenu: {e}")
            print("Le contenu original sera utilisé.")
            return content
    
    def save_document(self, content, output_path):
        """
        Enregistre le contenu dans un fichier.
        
        Args:
            content: Contenu Markdown à enregistrer
            output_path: Chemin du fichier de sortie
            
        Returns:
            bool: True si l'enregistrement a réussi, False sinon
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"⚠️ Erreur lors de l'écriture du fichier: {e}")
            return False
    
    # Méthode _extract_sections_from_instructions déplacée vers lib/creator.py

def main():
    """
    Fonction principale pour l'exécution de l'outil en ligne de commande avec interface interactive.
    """
    parser = argparse.ArgumentParser(
        description="CreateDoc - Créer et améliorer un document Markdown structuré"
    )
    parser.add_argument(
        "-o", "--output",
        help="Chemin vers le fichier de sortie (par défaut: titre_document.md)"
    )
    parser.add_argument(
        "--api-key",
        help="Clé API pour le modèle LLM (si non défini dans .env)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Active le mode debug pour plus d'informations"
    )
    parser.add_argument(
        "--model",
        help="Nom du modèle LLM à utiliser (par défaut: défini dans .env ou gpt-3.5-turbo)"
    )
    parser.add_argument(
        "--no-improve",
        action="store_true",
        help="Désactive l'amélioration automatique du document"
    )
    
    args = parser.parse_args()
    
    # Configurer le mode debug si demandé
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        print("Mode debug activé")
    
    # Interface interactive pour recueillir les entrées utilisateur
    print("📝 Bienvenue dans CreateDoc 📝")
    print("-------------------------------")
    
    # Demander le titre du document - élément principal qui détermine le contenu
    title = input("Entrez le titre du document : ")
    while not title.strip():
        title = input("Le titre ne peut pas être vide. Veuillez entrer un titre : ")
    
    # Demander le sujet principal (facultatif) - orientation thématique
    print("\nEntrez le sujet principal du document (facultatif).")
    print("Le sujet permet d'orienter le contenu, par exemple 'Intelligence Artificielle'")
    print("pour un document sur 'Google' axera le contenu sur les contributions de Google en IA.")
    subject = input("Sujet (appuyez sur Entrée pour ignorer) : ")
    
    # Demander les instructions
    print("\nEntrez les instructions spécifiques pour le contenu (facultatif).")
    print("Exemples:")
    print("- Niveau de lecture: débutant, intermédiaire, expert")
    print("- Mots-clés comme 'histoire', 'applications', 'défis', qui deviendront des sections")
    print("- Cadre temporel ou géographique")
    instructions = input("Instructions (appuyez sur Entrée pour ignorer) : ")
    
    # Déterminer le chemin du fichier de sortie
    if args.output:
        output_path = Path(args.output)
    else:
        # Créer un nom de fichier à partir du titre (en remplaçant les espaces par des underscores)
        safe_title = title.replace(' ', '_').lower()
        output_path = Path(f"{safe_title}.md")
    
    # Créer le service de création de documents
    document_creator = DocumentCreator(api_key=args.api_key, model=args.model)
    
    # Générer le contenu du document
    print(f"\n📝 Création du document '{title}' en cours...")
    content = document_creator.create_document(
        title=title,
        subject=subject,
        instructions=instructions
    )
    
    # Enregistrer le document de base
    base_output_path = output_path
    if document_creator.save_document(content, base_output_path):
        print(f"✅ Document de base enregistré dans {base_output_path}")
    
    # Demander à l'utilisateur s'il souhaite améliorer le document
    if not args.no_improve:
        improve = input("\nSouhaitez-vous améliorer le document avec des informations de Wikipedia ? (o/n) : ").lower()
        if improve == 'o' or improve == 'oui':
            print(f"🔍 Amélioration du document en cours...")
            improved_content = document_creator.improve_document(content)
            
            # Déterminer le chemin du fichier amélioré
            improved_output_path = base_output_path.parent / f"{base_output_path.stem}_improved{base_output_path.suffix}"
            
            # Enregistrer le document amélioré
            if document_creator.save_document(improved_content, improved_output_path):
                print(f"✅ Document amélioré enregistré dans {improved_output_path}")
        else:
            print("➡️ Amélioration ignorée sur demande de l'utilisateur.")
    
    print("\n✅ Processus terminé avec succès!")

if __name__ == "__main__":
    main()