# ImproveDoc

**ImproveDoc** est un outil Python permettant de créer et d'améliorer des documents Markdown structurés grâce à l'IA et l'enrichissement automatique via Wikipédia.

## Fonctionnalités

- **Génération interactive** de documents Markdown à partir d'un titre, d'un sujet et d'instructions personnalisées
- **Amélioration automatique** de fichiers Markdown existants avec des informations et des liens issus de Wikipédia
- **Agents spécialisés** pour la rédaction, l'enrichissement, la vérification et le formatage
- **Mémoire avancée** basée sur Chroma pour éviter les redondances et personnaliser les ajouts
- **Embedder multilingue** basé sur SentenceTransformers
- **Configuration fine** de l'API LLM, la langue wiki, et d'autres options via fichiers YAML ou la CLI

## Installation

1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/iapourtous/improveDoc.git
   cd improveDoc
   ```

2. **Installer l'environnement virtuel :**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Installer le package en mode développement :**
   ```bash
   # Installe le package et toutes ses dépendances
   pip install -e .
   ```

4. **Configurer l'environnement :**
   ```bash
   cp .env.example .env
   # Éditer .env pour renseigner votre clé API, modèle LLM, langue, etc.
   mkdir -p improvedoc_memory
   ```

## Utilisation

### Créer un nouveau document Markdown

```bash
# Après installation avec pip install -e .
python -m improvedoc.create_doc
```

* Suivez l'interface interactive pour spécifier le titre, le sujet et les instructions

### Améliorer un document existant

```bash
python -m improvedoc.improvedoc chemin/vers/mon_fichier.md
```

* Le résultat est enregistré dans `<nom>_improved.md`

### Utilisation sans installation (pour développement)

Si vous préférez ne pas installer le package, vous pouvez utiliser PYTHONPATH :

```bash
PYTHONPATH=src python src/improvedoc/create_doc.py
PYTHONPATH=src python src/improvedoc/improvedoc.py chemin/vers/mon_fichier.md
```

### Options CLI utiles
* `--api-key` : Clé API LLM (prend le dessus sur .env)
* `--model` : Choix du modèle (par défaut: valeur .env ou gpt-3.5-turbo)
* `--debug` : Mode trace
* `--no-improve` : (create_doc.py) Désactive l'amélioration automatique
* `-o, --output` : Spécifier le fichier de sortie

## Exemples

* `exemple.md` et `exemple2.md` : Documents générés de base
* `exemple_improved.md` et `exemple2_improved.md` : Versions améliorées automatiquement

## Architecture du projet

```
improveDoc/
│
├── src/
│   ├── improvedoc/            # Package principal
│   │   ├── __init__.py
│   │   ├── improvedoc.py      # amélioration de document
│   │   └── create_doc.py      # création interactive
│   └── lib/                   # bibliothèques partagées
│
├── config/yaml/
│      ├── agents.yaml         # configuration des agents
│      └── defaults.yaml       # paramètres par défaut
│
├── improvedoc_memory/         # stockage des embeddings
├── tests/                     # tests unitaires
├── pyproject.toml            # configuration du package et dépendances
└── README.md
```

## Structure des imports

Le projet suit une structure modulaire avec le code source dans `src/`:

- Les scripts dans `src/improvedoc/` importent des modules depuis `lib/`
- L'installation en mode développement (`pip install -e .`) ou l'utilisation de `PYTHONPATH=src` est nécessaire pour résoudre correctement ces imports

## Configuration

La configuration se fait via les fichiers YAML dans `config/yaml/` ou par variables d'environnement dans `.env` :

* `CREW_API_KEY` : Clé API LLM (OpenAI, OpenRouter, etc.)
* `CREW_MODEL` : Nom du modèle par défaut
* `WIKI_LANGUAGE` : `fr`/`en`/`es`/etc, langue Wiki
* `CREWAI_STORAGE_DIR` : Répertoire pour la mémoire

## Dépendances principales

Toutes les dépendances sont spécifiées dans le fichier `pyproject.toml`:

* [CrewAI](https://github.com/joaomdmoura/crewAI) : orchestrateur d'agents
* [wikipedia](https://pypi.org/project/wikipedia/) : extraction de contenu Wiki
* [chromadb](https://docs.trychroma.com/) : stockage de mémoire à court terme
* [sentence-transformers](https://www.sbert.net/) : Embedder multilingue
* [python-dotenv](https://pypi.org/project/python-dotenv/) : gestion de configuration

## Tests

Des tests automatisés sont disponibles dans le dossier `tests/`. Pour les exécuter :

```bash
pytest
```

## Contribution

Contributions bienvenues : PR, suggestions de doc, amélioration de l'architecture, nouveaux agents ou outils !

## Licence

Projet distribué sous licence MIT.