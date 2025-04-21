#!/usr/bin/env python3
"""
ImproveDoc - Outil d'amélioration de contenu Markdown avec Wikipedia et CrewAI

Cet outil prend en entrée un fichier Markdown et produit une version améliorée 
en enrichissant le contenu avec des informations provenant de Wikipedia.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
import warnings

# Ajouter le répertoire courant au chemin pour permettre les imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Désactiver les warnings
warnings.filterwarnings("ignore", category=Warning)

try:
    from lib.enhancer import ContentEnhancer
    
    # Vérifier que les modules nécessaires sont installés
    import crewai
    import wikipedia
except ImportError as e:
    print(f"⚠️ Erreur d'importation: {e}")
    print("⚠️ Certaines dépendances requises ne sont pas installées.")
    print("Installez-les avec : pip install -r requirements.txt")
    print("Assurez-vous d'activer l'environnement virtuel : source venv/bin/activate")
    sys.exit(1)

def main():
    """
    Fonction principale pour l'exécution de l'outil en ligne de commande.
    """
    parser = argparse.ArgumentParser(
        description="ImproveDoc - Améliore un document Markdown avec des informations de Wikipedia"
    )
    parser.add_argument(
        "input_file",
        help="Chemin vers le fichier Markdown à améliorer"
    )
    parser.add_argument(
        "-o", "--output",
        help="Chemin vers le fichier de sortie (par défaut: nom_du_fichier_improved.md)"
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
    
    args = parser.parse_args()
    
    # Vérifier que le fichier d'entrée existe
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"⚠️ Erreur: Le fichier {input_path} n'existe pas.")
        sys.exit(1)
    
    # Déterminer le chemin du fichier de sortie
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_improved{input_path.suffix}"
    
    # Lire le contenu du fichier d'entrée
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"⚠️ Erreur lors de la lecture du fichier: {e}")
        sys.exit(1)
    
    # Configurer le mode debug si demandé
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        print("Mode debug activé")
    
    # Améliorer le contenu
    print(f"🔍 Analyse et amélioration du contenu en cours...")
    try:
        # Créer le service d'amélioration de contenu
        enhancer = ContentEnhancer(api_key=args.api_key, model=args.model)
        
        # Améliorer le contenu
        improved_content = enhancer.enhance_content(content)
        
        # Vérifier si le contenu a été réellement amélioré
        if improved_content == content:
            print("⚠️ Le contenu n'a pas pu être amélioré en raison d'erreurs avec l'API LLM.")
            print("Essayez avec une clé API valide (paramètre --api-key ou variable d'environnement CREW_API_KEY).")
        
        # Écrire le contenu dans le fichier de sortie
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(improved_content)
            print(f"✅ Contenu enregistré dans {output_path}")
        except Exception as e:
            print(f"⚠️ Erreur lors de l'écriture du fichier: {e}")
    except ValueError as e:
        print(f"⚠️ Erreur de configuration: {e}")
    except Exception as e:
        print(f"⚠️ Erreur lors de l'amélioration du contenu: {e}")
        print("Le fichier d'entrée n'a pas été modifié.")

if __name__ == "__main__":
    main()