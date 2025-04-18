# ImproveDoc

**ImproveDoc** est un outil Python permettant de créer et d'améliorer des documents Markdown structurés grâce à l'intelligence artificielle (CrewAI) et à l'enrichissement automatique via Wikipedia.

## Fonctionnalités

- Génération de documents Markdown structurés à partir d'un titre, d'un sujet et d'instructions personnalisées (`createDoc.py`)
- Amélioration automatique de documents Markdown existants avec des informations issues de Wikipedia (`improvedoc.py`)
- Agents spécialisés pour la rédaction, la vérification, l'enrichissement et l'édition de contenu (`lib/agents.py`)
- Personnalisation du modèle LLM, de la langue Wikipedia et des paramètres via `.env`

## Installation

1. Clonez ce dépôt :
   ```sh
   git clone <repo-url>
   cd improveDoc
   ```

2. Installez les dépendances :
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Configurez votre clé API dans `.env` (voir `.env.example`).

## Utilisation

### Créer un document Markdown

```sh
python createDoc.py
```
Suivez l'interface interactive pour saisir le titre, le sujet et les instructions.

### Améliorer un document Markdown existant

```sh
python improvedoc.py <fichier.md>
```
Le résultat sera enregistré dans `<fichier>_improved.md`.

### Options avancées

- `--api-key` : Spécifier une clé API OpenAI/OpenRouter
- `--model` : Choisir le modèle LLM à utiliser
- `--debug` : Activer le mode debug

## Exemples

- `exemple.md` : Document de base généré
- `exemple_improved.md` : Version enrichie automatiquement

## Structure du projet

- `createDoc.py` : Génération interactive de documents
- `improvedoc.py` : Amélioration de documents existants
- `lib/` : Modules internes (agents, outils, configuration, traitement Markdown)

## Configuration

Voir le fichier `lib/config.py` et `.env.example` pour les variables d'environnement à définir.

## Dépendances principales

- [CrewAI](https://github.com/joaomdmoura/crewAI)
- [wikipedia](https://pypi.org/project/wikipedia/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)

## Licence

Ce projet est distribué sous licence MIT.