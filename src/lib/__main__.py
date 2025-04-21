"""
Point d'entrée pour l'exécution en tant que module.
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au chemin pour permettre les imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Importer et exécuter la fonction principale
from improvedoc import main

if __name__ == "__main__":
    main()