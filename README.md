# ImproveDoc

Un outil en ligne de commande pour améliorer des documents Markdown avec des informations issues de Wikipedia en utilisant la puissance de CrewAI et des agents d'IA.

## Fonctionnalités

- **Enrichissement automatique** : Ajoute des informations pertinentes issues de Wikipedia
- **Vérification factuelle** : Vérifie la véracité des affirmations dans le document
- **Ajout de liens** : Transforme les termes importants en liens vers les articles Wikipedia correspondants
- **Amélioration du format Markdown** : Assure une structure cohérente et lisible

## Installation

### Prérequis

- Python 3.8 ou supérieur
- Pip (gestionnaire de paquets Python)

### Installation des dépendances

```bash
pip install -r requirements.txt
```

## Configuration

Créez un fichier `.env` à la racine du projet avec les variables d'environnement suivantes :

```
OPENAI_API_KEY=votre_clé_api_openai
LLM_MODEL=gpt-3.5-turbo  # ou un autre modèle compatible
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4096
```

## Utilisation

```bash
python improvedoc.py chemin/vers/votre/fichier.md
```

### Options

- `-o, --output` : Spécifier le chemin du fichier de sortie (par défaut: nom_du_fichier_improved.md)
- `--api-key` : Fournir directement la clé API (plutôt que via le fichier .env)
- `--model` : Spécifier le modèle à utiliser (plutôt que celui défini dans .env)

### Exemples

```bash
# Amélioration simple avec fichier de sortie par défaut
python improvedoc.py document.md

# Spécifier un fichier de sortie personnalisé
python improvedoc.py document.md -o document_enrichi.md

# Utiliser une clé API spécifique et un modèle alternatif
python improvedoc.py document.md --api-key sk-xxxx --model gpt-4
```

## Comment ça marche

ImproveDoc utilise plusieurs agents spécialisés coordonnés par CrewAI :

1. **Agent Chercheur** : Recherche des informations pertinentes sur Wikipedia pour enrichir le contenu
2. **Agent Vérificateur de Faits** : Vérifie la véracité des affirmations dans le document
3. **Agent Spécialiste des Liens** : Ajoute des liens vers les articles Wikipedia correspondants
4. **Agent Éditeur Markdown** : Assure une structure cohérente et lisible

Ces agents travaillent séquentiellement sur chaque section du document pour produire un contenu enrichi et bien structuré.

## Licence

MIT

## Auteur

[Votre nom]

## Contributions

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.