"""
Agents pour ImproveDoc et CreateDoc.

Ce module contient les classes pour créer les différents agents CrewAI.
"""

# Standard library
from typing import List, Dict
import os

# Third‑party
import yaml

# CrewAI
from crewai import Agent, LLM

from .tools import (
    search_wikipedia, get_wikipedia_summary, 
    get_wikipedia_content, get_wikipedia_url, 
    set_wikipedia_language
)

class AgentFactory:
    """
    Usine à agents pour ImproveDoc et CreateDoc.
    """
    
    def __init__(self, llm: LLM, wiki_language: str = "fr"):
        """
        Initialise la factory d'agents.
        
        Args:
            llm: Instance de LLM à utiliser par les agents
            wiki_language: Langue pour les recherches Wikipedia
        """
        self.llm = llm
        
        # Configurer la langue de Wikipedia
        set_wikipedia_language(wiki_language)

        # ------------------------------------------------------------------
        # Chargement de la configuration YAML des agents
        # ------------------------------------------------------------------

        self._agents_cfg: Dict[str, Dict] = {}
        cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),  # <repo_root>/config/yaml
            "config",
            "yaml",
            "agents.yaml",
        )

        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    self._agents_cfg = yaml.safe_load(f) or {}
            except Exception as e:  # pragma: no cover – échec de YAML n'est pas bloquant
                print(f"⚠️ Impossible de parser {cfg_path}: {e}")

        # ------------------------------------------------------------------
        # Mutualisation des listes d'outils Wikipedia
        # ------------------------------------------------------------------
        # Les listes d'outils CrewAI sont des séquences d'objets ``Tool``.
        # Comme ils sont immuables et idempotents nous pouvons les préparer
        # une seule fois afin d'éviter de recréer la même liste pour chaque
        # agent et ainsi réduire la consommation mémoire globale.

        self._wiki_full_toolkit = [
            search_wikipedia,
            get_wikipedia_summary,
            get_wikipedia_content,
            get_wikipedia_url,
        ]

        self._wiki_link_toolkit = [
            search_wikipedia,
            get_wikipedia_url,
        ]

    # ------------------------------------------------------------------
    # Méthodes utilitaires internes
    # ------------------------------------------------------------------

    def _get_yaml_cfg(self, key: str) -> Dict[str, str]:
        """Renvoie le dictionnaire de configuration YAML pour *key* (ou {})."""
        return self._agents_cfg.get(key, {}) if self._agents_cfg else {}

    def _build_agent(
        self,
        key: str,
        *,
        fallback_role: str,
        fallback_goal: str,
        fallback_backstory: str,
        tools: List,
        memory: bool = False,
        llm: LLM | None = None,
    ) -> Agent:
        """Fabrique un Agent CrewAI à partir du YAML, avec valeurs de repli.

        Cette fonction centralise la lecture de la configuration YAML afin
        d’éviter la duplication de code dans chaque *create_*.
        """
        cfg = self._get_yaml_cfg(key)

        role = cfg.get("role", fallback_role)
        goal = cfg.get("goal", fallback_goal)
        backstory = cfg.get("backstory", fallback_backstory)

        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=tools,
            allow_delegation=False,
            verbose=True,
            llm=llm or self.llm,
            memory=memory,
        )
    
    def create_research_agent(self) -> Agent:
        """
        Crée un agent de recherche pour enrichir le contenu avec des informations de Wikipédia.
        
        Returns:
            Agent CrewAI pour la recherche et l'enrichissement
        """
        # Utilisation d'une liste d'outils mutualisée afin d'éviter la création
        # répétitive d'instances identiques pour chaque appel. Ceci réduit la
        # surcharge mémoire induite par CrewAI lorsque plusieurs agents
        # partagent exactement les mêmes outils.
        tools = self._wiki_full_toolkit

        return self._build_agent(
            "research_agent",
            fallback_role="Chercheur Wikipédia",
            fallback_goal="Enrichir le contenu avec des informations pertinentes et factuelles de Wikipédia",
            fallback_backstory=(
                "Tu es un expert en recherche documentaire capable d'extraire "
                "des informations précises et pertinentes de Wikipédia pour enrichir un contenu existant. "
                "Tu sais identifier les lacunes informatives dans un texte et les combler avec "
                "des données fiables. Tu maîtrises l'art d'intégrer harmonieusement de nouvelles "
                "informations tout en préservant la structure et le style du document original. "
                "Tu utilises ta mémoire pour te souvenir des recherches précédentes et éviter les "
                "redondances dans tes recherches et enrichissements."
            ),
            tools=tools,
            memory=True,
        )
    
    def create_fact_checker_agent(self) -> Agent:
        """
        Crée un agent de vérification des faits.
        
        Returns:
            Agent CrewAI pour la vérification factuelle
        """
        tools = self._wiki_full_toolkit

        return self._build_agent(
            "fact_checker_agent",
            fallback_role="Vérificateur de Faits",
            fallback_goal="Vérifier la véracité des affirmations présentes dans le contenu",
            fallback_backstory=(
                "Tu es un fact-checker méticuleux qui vérifie scrupuleusement "
                "chaque affirmation importante. Tu compares les informations avec des sources "
                "fiables comme Wikipédia pour confirmer leur exactitude. Tu sais détecter les "
                "inexactitudes, les simplifications excessives et les affirmations non étayées. "
                "Tu proposes des corrections précises basées uniquement sur des sources vérifiables."
            ),
            tools=tools,
            memory=False,
        )
    
    def create_wiki_linker_agent(self) -> Agent:
        """
        Crée un agent pour l'ajout de liens Wikipédia pertinents.
        
        Returns:
            Agent CrewAI pour l'ajout de liens
        """
        tools = self._wiki_link_toolkit

        return self._build_agent(
            "wiki_linker_agent",
            fallback_role="Spécialiste des Liens Wikipédia",
            fallback_goal="Identifier et lier les termes pertinents vers des articles Wikipédia",
            fallback_backstory=(
                "Tu es un expert en hypertexte et en organisation de contenu. "
                "Tu excelles dans l'identification des termes, concepts et entités qui méritent "
                "d'être liés à leurs articles Wikipédia correspondants. Tu sais distinguer les "
                "termes vraiment importants des termes secondaires, et tu places les liens de "
                "manière stratégique pour enrichir la navigation sans surcharger le texte. "
                "Tu utilises ta mémoire pour te souvenir des termes déjà liés dans d'autres "
                "sections, assurant ainsi une cohérence dans tout le document. Tu évites de "
                "créer des liens redondants pour les mêmes concepts et tu préfères lier des "
                "termes différents pour diversifier les possibilités de navigation."
            ),
            tools=tools,
            memory=True,
        )
    
    def create_markdown_editor_agent(self) -> Agent:
        """
        Crée un agent d'édition pour assurer la qualité du Markdown.
        
        Returns:
            Agent CrewAI pour l'édition Markdown
        """
        return self._build_agent(
            "markdown_editor_agent",
            fallback_role="Éditeur Markdown",
            fallback_goal="Garantir une structure Markdown propre et cohérente",
            fallback_backstory=(
                "Tu es un expert en édition et formatage Markdown. Tu veilles "
                "à la cohérence structurelle et visuelle des documents, en t'assurant que "
                "la hiérarchie des titres est logique, que les listes sont correctement "
                "formatées, et que la présentation générale est optimale. Tu préserves "
                "soigneusement le style original tout en améliorant la lisibilité et "
                "l'organisation du contenu. Tu utilises ta mémoire pour te souvenir des "
                "décisions de formatage prises précédemment, assurant ainsi une cohérence "
                "stylistique dans l'ensemble du document. Tu veilles à ce que les conventions "
                "de formatage soient appliquées de manière uniforme à travers tout le document."
            ),
            tools=[],
            memory=True,
        )
    
    def create_content_writer_agent(self, llm=None) -> Agent:
        """
        Crée un agent rédacteur de contenu pour CreateDoc.
        
        Args:
            llm: Un LLM optionnel pour cet agent (utilise l'instance par défaut sinon)
            
        Returns:
            Agent CrewAI pour la rédaction de contenu
        """
        cfg_key = "content_writer_agent"

        return self._build_agent(
            cfg_key,
            fallback_role="Rédacteur de Contenu",
            fallback_goal="Rédiger un document Markdown structuré, informatif et adapté à l'audience cible",
            fallback_backstory=(
                "Tu es un rédacteur professionnel spécialisé dans la création "
                "de contenu pédagogique et informatif. Ton expertise te permet de rédiger "
                "des documents clairs, structurés et adaptés à différents niveaux d'audience. "
                "Tu maîtrises la rédaction en format Markdown et sais organiser l'information "
                "de manière logique et progressive, en créant un contenu substantiel "
                "et informatif pour chaque section."
            ),
            tools=[],
            memory=False,
            llm=llm,
        )
    
    def create_editor_chief_agent(self, llm=None) -> Agent:
        """
        Crée un agent chef de rédaction pour CreateDoc.
        
        Args:
            llm: Un LLM optionnel pour cet agent (utilise l'instance par défaut sinon)
            
        Returns:
            Agent CrewAI chef de rédaction
        """
        return self._build_agent(
            "editor_chief_agent",
            fallback_role="Chef de Rédaction",
            fallback_goal="Vérifier et améliorer le contenu pour assurer sa qualité et le respect des contraintes",
            fallback_backstory=(
                "Tu es un chef de rédaction expérimenté, attentif aux détails et rigoureux. "
                "Ton rôle est de garantir que le contenu respecte les consignes précises "
                "et maintient un haut niveau de qualité. Tu vérifies scrupuleusement la structure, "
                "la cohérence, et la clarté des documents, en t'assurant que chaque section "
                "contient un contenu substantiel et informatif. Tu corriges et améliores le "
                "texte sans hésiter quand c'est nécessaire pour garantir l'excellence du résultat final."
            ),
            tools=[],
            memory=False,
            llm=llm,
        )