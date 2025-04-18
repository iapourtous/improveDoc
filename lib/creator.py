"""
Module de création de documents Markdown.

Ce module contient les classes pour la création et la génération de documents Markdown
structurés à partir d'un titre, d'un sujet et d'instructions.
"""

import os
from typing import List, Optional
from crewai import Task, Crew, Process, LLM
import logging

from .agents import AgentFactory
from .config import Config

class ContentCreator:
    """
    Service de création de documents Markdown structurés.
    """
    
    def __init__(self, agent_factory: AgentFactory, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialise le créateur de contenu.
        
        Args:
            agent_factory: Une instance de AgentFactory pour créer les agents
            api_key: Clé API optionnelle pour le modèle LLM
            model: Nom du modèle LLM optionnel
        """
        # Initialiser le logger d'abord pour pouvoir l'utiliser
        self.logger = logging.getLogger(__name__)
        
        # Charger la configuration
        self.config = Config(api_key=api_key, model=model)
        
        # Vérifier que la configuration est valide
        if not self.config.validate():
            self.logger.warning("Configuration invalide. Le contenu de base sera utilisé.")
            
        self.agent_factory = agent_factory
    
    def create_content(self, title: str, subject: str = "", instructions: str = "") -> str:
        """
        Crée un document Markdown structuré en utilisant des agents spécialisés.
        
        Args:
            title: Titre du document
            subject: Sujet principal (orientation thématique, facultatif)
            instructions: Instructions spécifiques (facultatif)
            
        Returns:
            str: Contenu Markdown généré
        """
        self.logger.info(f"Création de contenu pour '{title}'")
        
        # Analyser les instructions pour extraire les sections
        sections = self._extract_sections_from_instructions(instructions)
        
        # Définir le niveau d'audience à partir des instructions
        audience = self._determine_audience(instructions)
        
        try:
            # Créer les agents
            writer_agent = self.agent_factory.create_content_writer_agent()
            editor_agent = self.agent_factory.create_editor_chief_agent()
            
            # Créer les tâches
            write_task = self._create_writing_task(
                writer_agent, title, subject, sections, audience, instructions
            )
            
            edit_task = self._create_editing_task(
                editor_agent, title, subject, sections, audience, instructions, write_task
            )
            
            # Créer et exécuter l'équipe
            crew = Crew(
                agents=[writer_agent, editor_agent],
                tasks=[write_task, edit_task],
                verbose=False,
                process=Process.sequential
            )
            
            # Exécuter et récupérer le résultat
            result = crew.kickoff()
            
            # Vérifier que le résultat est un document Markdown valide
            if result:
                content = str(result)
                
                # Nettoyage potentiel des artefacts d'agent et balises markdown
                if content.startswith("# Agent:") or content.startswith("## Final Answer:"):
                    self.logger.info("Nettoyage des artefacts d'agent dans le document")
                    # Essayer de trouver le début du vrai document Markdown
                    real_md_start = content.find("# ")
                    if real_md_start >= 0:
                        content = content[real_md_start:]
                
                # Supprimer les balises de code si présentes (```markdown)
                # Faire un remplacement plus sélectif pour ne pas perturber le document
                content = content.replace("```markdown", "")
                # Ne pas supprimer toutes les balises ``` car cela pourrait affecter l'indentation
                # et la structure du document
                
                if self._is_valid_markdown(content):
                    return content
                else:
                    self.logger.warning("Le résultat n'est pas un document Markdown valide")
                    return self._create_fallback_document(title, subject, sections)
            else:
                self.logger.warning("Aucun résultat obtenu")
                return self._create_fallback_document(title, subject, sections)
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la création du document: {e}")
            return self._create_fallback_document(title, subject, sections)
    
    def _create_writing_task(self, writer_agent, title, subject, sections, audience, instructions) -> Task:
        """
        Crée la tâche de rédaction pour l'agent rédacteur.
        """
        sections_list = ', '.join([f'- {section}' for section in sections])
        
        # Préparer les variables avec apostrophes pour éviter les problèmes avec f-strings
        subject_orientation = 'Oriente ton contenu vers l\'aspect: ' + subject if subject else ''
        content_orientation = 'Oriente le contenu autour de l\'aspect: ' + subject if subject else 'Couvre les aspects les plus importants du sujet'
        
        # Préparer les suggestions de sections sans utiliser de f-string avec \n
        if sections_list:
            sections_suggestion = 'Voici quelques idées de sections suggérées (tu es entièrement libre de les adapter ou d\'en choisir d\'autres plus pertinentes):\n' + sections_list
        else:
            sections_suggestion = 'Tu es libre de structurer le document avec les sections que tu jugeras les plus pertinentes pour le sujet.'
        
        description = f"""
        Rédige un document Markdown structuré sur "{title}".
        
        {subject_orientation}
        
        Structure générale du document:
        - Un titre principal avec "# {title}"
        - Une organisation claire et logique avec plusieurs sections de niveau ##
        - Une bonne couverture du sujet
        
        {sections_suggestion}
        
        Directives pour le contenu:
        - Le document doit être centré sur "{title}" comme sujet principal
        - {content_orientation}
        - Rédige un contenu SUBSTANTIEL avec plusieurs phrases informatives par section
        - Adapte le contenu pour un public de niveau {audience}
        - Utilise tes connaissances pour produire un document informatif et exact
        
        Directives pour le format:
        - Structure claire avec titre principal (niveau #) et sections (niveau ##)
        - Pas de sous-sections (pas de ###)
        - Format Markdown correct et bien présenté
        - N'UTILISE PAS de blocs de code avec ```markdown
        
        Instructions spécifiques: {instructions if instructions else 'Aucune instruction spécifique.'}
        
        Assure-toi que chaque section contient des informations pertinentes et intéressantes.
        """
        
        return Task(
            description=description,
            agent=writer_agent,
            expected_output=f"Document Markdown bien structuré sur {title}"
        )
    
    def _create_editing_task(self, editor_agent, title, subject, sections, audience, instructions, write_task) -> Task:
        """
        Crée la tâche d'édition pour l'agent chef de rédaction.
        """
        sections_format = ', '.join([f'## {section}' for section in sections])
        
        subject_constraint = f'- Le contenu est orienté vers l\'aspect "{subject}"' if subject else ''
        description = f"""
        Révise le document Markdown créé par le rédacteur.
        
        Tes responsabilités:
        1. Vérifier que le document a une structure Markdown adéquate:
           - Un titre principal (niveau #) en rapport avec "{title}"
           - Plusieurs sections (niveau ##) bien organisées
           - Un bon équilibre entre les sections
           - PAS de sous-sections (pas de ###)
        
        2. Vérifier la qualité du contenu:
           - Le contenu est centré sur "{title}" comme sujet principal
           {subject_constraint}
           - Le contenu est adapté à un public de niveau {audience}
           - Chaque section contient un contenu SUBSTANTIEL (minimum 3-4 phrases)
           - Le formatage Markdown est correct
           
        3. Améliorer:
           - La clarté et la concision du texte
           - La cohérence entre les sections
           - La qualité de l'information
           - L'organisation logique du contenu
        
        4. Vérifier le respect des instructions spécifiques: {instructions if instructions else 'Aucune instruction spécifique.'}
        
        IMPORTANT: Si le document ne respecte pas les contraintes, tu dois le corriger complètement.
        Retourne le document Markdown complet et amélioré, pas juste des suggestions.
        """
        
        return Task(
            description=description,
            agent=editor_agent,
            context=[write_task],
            expected_output=f"Document Markdown final sur {title} conforme aux exigences"
        )
    
    def _extract_sections_from_instructions(self, instructions: str) -> List[str]:
        """
        Extrait les sections potentielles à partir des instructions.
        Ces sections ne sont que des suggestions pour l'agent, qui est libre de les ignorer.
        
        Args:
            instructions: Instructions fournies par l'utilisateur
            
        Returns:
            list: Liste des suggestions de sections (peut être vide)
        """
        # Version simple pour extraire les sections
        sections = []
        
        # Rechercher des mots-clés courants qui pourraient être des sections
        keywords = ["histoire", "contexte", "applications", "utilisations", 
                   "avantages", "inconvénients", "défis", "futur", 
                   "technologies", "méthodes", "exemples", "études de cas"]
        
        # Vérifier chaque mot-clé dans les instructions
        if instructions:
            for keyword in keywords:
                if keyword.lower() in instructions.lower():
                    # Capitaliser le premier caractère
                    formatted_keyword = keyword[0].upper() + keyword[1:]
                    sections.append(formatted_keyword)
        
        # Ne pas ajouter de sections par défaut pour laisser l'agent complètement libre
        # si aucune suggestion n'est extraite des instructions
        return sections
    
    def _determine_audience(self, instructions: str) -> str:
        """
        Détermine le niveau d'audience à partir des instructions.
        
        Args:
            instructions: Instructions fournies par l'utilisateur
            
        Returns:
            str: Niveau d'audience ('tout public', 'débutant', 'technique', 'spécialisé')
        """
        audience = "tout public"
        
        if instructions:
            if "spécialisé" in instructions.lower() or "expert" in instructions.lower():
                audience = "spécialisé"
            elif "technique" in instructions.lower():
                audience = "technique"
            elif "débutant" in instructions.lower() or "novice" in instructions.lower():
                audience = "débutant"
        
        return audience
    
    def _is_valid_markdown(self, content: str) -> bool:
        """
        Vérifie si le contenu est un document Markdown valide.
        
        Args:
            content: Contenu Markdown à vérifier
            
        Returns:
            bool: True si le document est valide, False sinon
        """
        # Vérifications minimales de la validité du document Markdown
        if not content:
            return False
            
        # Vérifier que le contenu est bien une chaîne de caractères
        if not isinstance(content, str):
            self.logger.warning(f"Le contenu n'est pas une chaîne mais un {type(content)}")
            return False
        
        # Vérifier que le contenu n'est pas juste un artefact d'agent
        # comme parfois vu dans certaines réponses
        if content.startswith("# Agent:") or content.startswith("## Final Answer:"):
            self.logger.warning("Le contenu contient des artefacts d'agent")
            # Si le contenu contient un titre Markdown, considérons-le comme valide
            # mais nous le nettoierons plus tard
            if "# " in content:
                self.logger.info("Mais contient un titre Markdown, considéré comme valide")
                return True
            return False
                
        # Vérification minimale - juste un titre principal
        # Le reste de la structure sera vérifié par l'agent éditeur
        has_title = "# " in content
        
        return has_title
    
    def _create_fallback_document(self, title: str, subject: str, sections: List[str]) -> str:
        """
        Crée un document de secours en cas d'échec de l'approche principale.
        
        Args:
            title: Titre du document
            subject: Sujet principal
            sections: Liste des sections
            
        Returns:
            str: Document Markdown de base
        """
        self.logger.info("Création d'un document de secours")
        
        content = f"# {title}\n\n"
        
        # Ajouter une section d'introduction
        content += f"## Introduction\n\n"
        if subject:
            content += f"Ce document présente {title} en relation avec {subject}.\n\n"
        else:
            content += f"Ce document présente {title} et ses différents aspects.\n\n"
        
        # Ajouter les sections au document
        for section in sections:
            content += f"## {section}\n\n"
            if subject:
                content += f"Cette section aborde {section.lower()} en relation avec {subject}.\n\n"
            else:
                content += f"Cette section traite des aspects liés à {section.lower()} pour {title}.\n\n"
        
        # Ajouter une section de conclusion
        content += f"## Conclusion\n\n"
        if subject:
            content += f"Ce document a présenté {title} en relation avec {subject} et ses aspects principaux."
        else:
            content += f"Ce document a présenté {title} et ses aspects principaux."
        
        return content