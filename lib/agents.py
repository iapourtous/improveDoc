"""
Agents pour ImproveDoc et CreateDoc.

Ce module contient les classes pour créer les différents agents CrewAI.
"""

from typing import List
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
    
    def create_research_agent(self) -> Agent:
        """
        Crée un agent de recherche pour enrichir le contenu avec des informations de Wikipédia.
        
        Returns:
            Agent CrewAI pour la recherche et l'enrichissement
        """
        tools = [
            search_wikipedia, 
            get_wikipedia_summary, 
            get_wikipedia_content, 
            get_wikipedia_url
        ]
        
        return Agent(
            role="Chercheur Wikipédia",
            goal="Enrichir le contenu avec des informations pertinentes et factuelles de Wikipédia",
            backstory="""Tu es un expert en recherche documentaire capable d'extraire 
            des informations précises et pertinentes de Wikipédia pour enrichir un contenu existant.
            Tu sais identifier les lacunes informatives dans un texte et les combler avec
            des données fiables. Tu maîtrises l'art d'intégrer harmonieusement de nouvelles 
            informations tout en préservant la structure et le style du document original.""",
            tools=tools,
            allow_delegation=False,
            verbose=True,
            llm=self.llm
        )
    
    def create_fact_checker_agent(self) -> Agent:
        """
        Crée un agent de vérification des faits.
        
        Returns:
            Agent CrewAI pour la vérification factuelle
        """
        tools = [
            search_wikipedia, 
            get_wikipedia_summary, 
            get_wikipedia_content, 
            get_wikipedia_url
        ]
        
        return Agent(
            role="Vérificateur de Faits",
            goal="Vérifier la véracité des affirmations présentes dans le contenu",
            backstory="""Tu es un fact-checker méticuleux qui vérifie scrupuleusement 
            chaque affirmation importante. Tu compares les informations avec des sources
            fiables comme Wikipédia pour confirmer leur exactitude. Tu sais détecter les
            inexactitudes, les simplifications excessives et les affirmations non étayées.
            Tu proposes des corrections précises basées uniquement sur des sources vérifiables.""",
            tools=tools,
            allow_delegation=False,
            verbose=True,
            llm=self.llm
        )
    
    def create_wiki_linker_agent(self) -> Agent:
        """
        Crée un agent pour l'ajout de liens Wikipédia pertinents.
        
        Returns:
            Agent CrewAI pour l'ajout de liens
        """
        tools = [
            search_wikipedia, 
            get_wikipedia_url
        ]
        
        return Agent(
            role="Spécialiste des Liens Wikipédia",
            goal="Identifier et lier les termes pertinents vers des articles Wikipédia",
            backstory="""Tu es un expert en hypertexte et en organisation de contenu. 
            Tu excelles dans l'identification des termes, concepts et entités qui méritent
            d'être liés à leurs articles Wikipédia correspondants. Tu sais distinguer les
            termes vraiment importants des termes secondaires, et tu places les liens de
            manière stratégique pour enrichir la navigation sans surcharger le texte.""",
            tools=tools,
            allow_delegation=False,
            verbose=True,
            llm=self.llm
        )
    
    def create_markdown_editor_agent(self) -> Agent:
        """
        Crée un agent d'édition pour assurer la qualité du Markdown.
        
        Returns:
            Agent CrewAI pour l'édition Markdown
        """
        return Agent(
            role="Éditeur Markdown",
            goal="Garantir une structure Markdown propre et cohérente",
            backstory="""Tu es un expert en édition et formatage Markdown. Tu veilles
            à la cohérence structurelle et visuelle des documents, en t'assurant que
            la hiérarchie des titres est logique, que les listes sont correctement
            formatées, et que la présentation générale est optimale. Tu préserves
            soigneusement le style original tout en améliorant la lisibilité et
            l'organisation du contenu.""",
            allow_delegation=False,
            verbose=True,
            llm=self.llm
        )
    
    def create_content_writer_agent(self, llm=None) -> Agent:
        """
        Crée un agent rédacteur de contenu pour CreateDoc.
        
        Args:
            llm: Un LLM optionnel pour cet agent (utilise l'instance par défaut sinon)
            
        Returns:
            Agent CrewAI pour la rédaction de contenu
        """
        return Agent(
            role="Rédacteur de Contenu",
            goal="Rédiger un document Markdown structuré, informatif et adapté à l'audience cible",
            backstory="""Tu es un rédacteur professionnel spécialisé dans la création 
            de contenu pédagogique et informatif. Ton expertise te permet de rédiger 
            des documents clairs, structurés et adaptés à différents niveaux d'audience.
            Tu maîtrises la rédaction en format Markdown et sais organiser l'information 
            de manière logique et progressive, en créant un contenu substantiel
            et informatif pour chaque section.""",
            allow_delegation=False,
            verbose=True,
            llm=llm or self.llm
        )
    
    def create_editor_chief_agent(self, llm=None) -> Agent:
        """
        Crée un agent chef de rédaction pour CreateDoc.
        
        Args:
            llm: Un LLM optionnel pour cet agent (utilise l'instance par défaut sinon)
            
        Returns:
            Agent CrewAI chef de rédaction
        """
        return Agent(
            role="Chef de Rédaction",
            goal="Vérifier et améliorer le contenu pour assurer sa qualité et le respect des contraintes",
            backstory="""Tu es un chef de rédaction expérimenté, attentif aux détails et rigoureux.
            Ton rôle est de garantir que le contenu respecte les consignes précises 
            et maintient un haut niveau de qualité. Tu vérifies scrupuleusement la structure, 
            la cohérence, et la clarté des documents, en t'assurant que chaque section 
            contient un contenu substantiel et informatif. Tu corriges et améliores le 
            texte sans hésiter quand c'est nécessaire pour garantir l'excellence du résultat final.""",
            allow_delegation=False,
            verbose=True,
            llm=llm or self.llm
        )