"""
Service principal d'amélioration de contenu Markdown pour ImproveDoc.

Ce module contient le service principal d'amélioration de contenu Markdown.
"""

import traceback
from typing import Dict, List, Optional

from crewai import Agent, Task, Crew, Process, LLM

# Imports relatifs pour les modules du package
from .config import Config
from .agents import AgentFactory
from .content_processor import MarkdownProcessor

class ContentEnhancer:
    """
    Service qui utilise CrewAI pour améliorer le contenu Markdown avec Wikipedia.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialise le service d'amélioration de contenu.
        
        Args:
            api_key: Clé API pour le modèle LLM (OpenAI, etc.)
            model: Nom du modèle à utiliser
        """
        # Chargement de la configuration
        self.config = Config(api_key=api_key, model=model)
        
        # Vérifier que la clé API est configurée
        if not self.config.validate():
            raise ValueError("Configuration invalide. Vérifiez la clé API.")
        
        # Configuration du LLM pour CrewAI
        try:
            # Essayer d'abord avec la configuration complète
            self.llm = LLM(
                model=self.config.model,
                api_key=self.config.api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                base_url=self.config.base_url,
            )
        except Exception as e:
            print(f"⚠️ Erreur lors de l'initialisation du LLM avec la configuration complète: {e}")
            print("Tentative d'initialisation avec une configuration simplifiée...")
            
            # Fallback sur une configuration simplifiée
            self.llm = LLM(
                model="openai/gpt-3.5-turbo"
            )
        
        # Création de la factory d'agents avec la langue spécifiée
        self.agent_factory = AgentFactory(self.llm, wiki_language=self.config.wikipedia_language)
        
        print(f"ContentEnhancer configuré avec modèle: {self.config.model}")
    
    def _create_enrichment_task(self, agent: Agent, section_name: str, section_content: str) -> Task:
        """
        Crée une tâche d'enrichissement pour une section spécifique.
        
        Args:
            agent: Agent qui effectuera la tâche
            section_name: Nom de la section
            section_content: Contenu de la section
            
        Returns:
            Tâche CrewAI
        """
        return Task(
            description=f"""Enrichir la section "{section_name}" avec des informations pertinentes de Wikipédia.
            
            CONTENU ACTUEL DE LA SECTION:
            ```
            {section_content}
            ```
            
            INSTRUCTIONS:
            1. Identifie les aspects qui pourraient bénéficier d'informations supplémentaires
            2. Recherche des informations pertinentes sur Wikipédia
            3. Intègre harmonieusement ces informations dans le contenu existant
            4. Conserve le style et le ton du contenu original
            5. Ajoute des détails factuels, des exemples ou des explications supplémentaires
            
            CONTRAINTES STRICTES:
            - Ne modifie PAS les listes à puces existantes, enrichis-les sans changer leur structure
            - N'ajoute que des informations factuelles et vérifiables
            - Maintiens la cohérence avec le reste du document
            - IMPORTANT: Préserve absolument tous les points importants du contenu original
            - Respecte la structure Markdown existante
            - NE CRÉE PAS de nouvelles sous-sections, enrichis uniquement le contenu existant
            - Si tu ajoutes des informations, ajoute-les en complément du contenu existant, pas en remplacement
            
            Fournis UNIQUEMENT le contenu de section enrichi, sans ajouter de titre de section.
            """,
            agent=agent,
            expected_output=f"Section '{section_name}' enrichie avec des informations de Wikipédia"
        )
    
    def _create_verification_task(self, agent: Agent, section_name: str, enrichment_task: Task) -> Task:
        """
        Crée une tâche de vérification factuelle pour une section.
        
        Args:
            agent: Agent qui effectuera la tâche
            section_name: Nom de la section
            enrichment_task: Tâche d'enrichissement dont le résultat sera vérifié
            
        Returns:
            Tâche CrewAI
        """
        return Task(
            description=f"""Vérifie la véracité des affirmations dans la section "{section_name}" enrichie.
            
            CONTENU DE LA SECTION ENRICHIE:
            ```
            {{{{enrichment_task.output}}}}
            ```
            
            INSTRUCTIONS:
            1. Identifie les affirmations factuelles importantes dans le texte
            2. Vérifie chaque affirmation par rapport à des sources fiables (Wikipédia)
            3. Corrige les informations inexactes ou douteuses
            4. Ajoute des nuances lorsque l'affirmation est trop catégorique
            
            CONTRAINTES:
            - Corrige uniquement les erreurs factuelles, pas les questions de style
            - Cite brièvement la source Wikipédia quand tu fais une correction majeure
            - Maintiens le flux logique du texte
            
            Fournis la section complète avec les corrections apportées.
            """,
            agent=agent,
            context=[enrichment_task],
            expected_output=f"Section '{section_name}' avec affirmations vérifiées"
        )
    
    def _create_linking_task(self, agent: Agent, section_name: str, verification_task: Task) -> Task:
        """
        Crée une tâche d'ajout de liens Wikipédia pour une section.
        
        Args:
            agent: Agent qui effectuera la tâche
            section_name: Nom de la section
            verification_task: Tâche de vérification dont le résultat recevra des liens
            
        Returns:
            Tâche CrewAI
        """
        return Task(
            description=f"""Ajoute des liens Wikipédia pertinents dans la section "{section_name}".
            
            CONTENU DE LA SECTION VÉRIFIÉE:
            ```
            {{{{verification_task.output}}}}
            ```
            
            INSTRUCTIONS:
            1. Identifie les termes, concepts et entités importants
            2. Recherche les articles Wikipédia correspondants
            3. Ajoute des liens Markdown au format [terme](https://fr.wikipedia.org/wiki/Page)
            4. Priorise les termes les plus significatifs et pertinents
            
            CONTRAINTES:
            - Ne modifie pas le contenu textuel, ajoute seulement des liens
            - Évite de lier des termes trop génériques
            - Ne lie chaque terme que lors de sa première occurrence importante
            - N'ajoute pas de liens dans les titres et sous-titres
            - Limite-toi à 3-5 liens pertinents par section
            
            Fournis la section complète avec les liens ajoutés.
            """,
            agent=agent,
            context=[verification_task],
            expected_output=f"Section '{section_name}' avec liens Wikipédia ajoutés"
        )
    
    def _create_editing_task(self, agent: Agent, linking_tasks: List[Task], original_structure: str) -> Task:
        """
        Crée une tâche d'édition Markdown globale.
        
        Args:
            agent: Agent qui effectuera la tâche
            linking_tasks: Liste des tâches d'ajout de liens dont les résultats seront édités
            original_structure: Description de la structure originale du document
            
        Returns:
            Tâche CrewAI
        """
        # Construire la description des sections
        sections_description = []
        for i, task in enumerate(linking_tasks):
            sections_description.append(f"Section {i+1}: {{{{linking_tasks[{i}].output}}}}")
        
        return Task(
            description=f"""Assure la qualité et la cohérence du format Markdown du document entier 
            en préservant EXACTEMENT la structure originale du document.
            
            STRUCTURE ORIGINALE DU DOCUMENT:
            ```
            {original_structure}
            ```
            
            SECTIONS AMÉLIORÉES AVEC LIENS:
            {chr(10).join(sections_description)}
            
            INSTRUCTIONS:
            1. Assure-toi que tous les titres originaux sont EXACTEMENT préservés (même niveau, même texte)
            2. Vérifie que les listes sont bien formatées
            3. Ajoute des sauts de ligne appropriés pour améliorer la lisibilité
            4. Harmonise le style de formatage dans tout le document
            
            CONTRAINTES STRICTES:
            - CRUCIAL: Préserve EXACTEMENT la structure de titres du document original
            - NE PAS ajouter de nouveaux titres ou sections qui n'existaient pas dans l'original
            - Préserve tout le contenu substantiel
            - Conserve les liens Wikipédia ajoutés
            - Ne modifie pas les informations factuelles importantes
            - NE modifie PAS l'ordre des sections
            
            Fournis le document Markdown complet correctement formaté qui respecte STRICTEMENT
            la structure originale.
            """,
            agent=agent,
            context=linking_tasks,
            expected_output="Document Markdown avec format amélioré préservant la structure originale"
        )
    
    def enhance_content(self, markdown_content: str) -> str:
        """
        Améliore un contenu Markdown avec des informations Wikipédia, vérification factuelle,
        et ajout de liens pertinents, tout en préservant la structure originale.
        
        Args:
            markdown_content: Contenu Markdown original à améliorer
            
        Returns:
            Contenu Markdown amélioré
        """
        # Afficher un message pour débug
        print(f"ContentEnhancer.enhance_content: Démarrage de l'amélioration du contenu")
        print(f"Longueur du contenu: {len(markdown_content)} caractères")
        
        # 1. Diviser le contenu en sections
        sections = MarkdownProcessor.parse_sections(markdown_content)
        
        # Créer une description de la structure originale pour référence
        original_structure = "\n".join([
            s.get('full_header', '') for _, s in sorted(
                [(k, v) for k, v in sections.items()], 
                key=lambda x: x[1].get('original_position', 999)
            ) if s.get('full_header')
        ])
        
        # Afficher les sections trouvées
        print(f"Sections trouvées: {len(sections)} sections")
        
        # 2. Créer les agents
        research_agent = self.agent_factory.create_research_agent()
        fact_checker_agent = self.agent_factory.create_fact_checker_agent()
        wiki_linker_agent = self.agent_factory.create_wiki_linker_agent()
        markdown_editor_agent = self.agent_factory.create_markdown_editor_agent()
        
        # 3. Générer dynamiquement les tâches pour chaque section
        enrichment_tasks = []
        verification_tasks = []
        linking_tasks = []
        
        # Ordonner les sections par leur position originale dans le document
        section_items = sorted(
            [(k, v) for k, v in sections.items()], 
            key=lambda x: x[1].get('original_position', 999)
        )
        
        # Limiter le nombre de sections traitées si nécessaire
        section_items = section_items[:5]  # Limiter à 5 sections pour la performance
        
        # Dictionnaire pour suivre la correspondance entre tâches et sections originales
        task_to_section_map = {}
        
        for section_id, section_data in section_items:
            section_name = section_data.get('title', 'Sans titre')
            section_content = section_data.get('content', '')
            
            if not section_content.strip():
                continue  # Ignorer les sections vides
            
            # Tâche d'enrichissement pour cette section
            enrichment_task = self._create_enrichment_task(
                research_agent, section_name, section_content
            )
            enrichment_tasks.append(enrichment_task)
            
            # Tâche de vérification factuelle (dépend de l'enrichissement)
            verification_task = self._create_verification_task(
                fact_checker_agent, section_name, enrichment_task
            )
            verification_tasks.append(verification_task)
            
            # Tâche d'ajout de liens (dépend de la vérification)
            linking_task = self._create_linking_task(
                wiki_linker_agent, section_name, verification_task
            )
            linking_tasks.append(linking_task)
            
            # Stocker la référence à la section d'origine dans un dictionnaire séparé
            task_to_section_map[id(linking_task)] = {
                "section_id": section_id,
                "section_data": section_data
            }
        
        # 4. Tâche d'édition Markdown globale (après que toutes les sections aient été traitées)
        markdown_editing_task = self._create_editing_task(
            markdown_editor_agent, linking_tasks, original_structure
        )
        
        # 5. Création et exécution de l'équipage
        crew = Crew(
            agents=[research_agent, fact_checker_agent, wiki_linker_agent, markdown_editor_agent],
            tasks=enrichment_tasks + verification_tasks + linking_tasks + [markdown_editing_task],
            verbose=True,
            process=Process.sequential
        )
        
        try:
            result = crew.kickoff()
            
            # Dans les nouvelles versions de CrewAI, crew.kickoff() retourne un objet CrewOutput
            # au lieu d'une chaîne de caractères, donc nous devons extraire le texte
            result_text = str(result) if hasattr(result, '__str__') else ""
            
            # Extraire le contenu final de la réponse
            markdown_content_improved = MarkdownProcessor.extract_final_content(result_text)
            
            # Pour préserver la structure en cas de problème avec le LLM
            if not markdown_content_improved or len(markdown_content_improved) < 50:
                print("⚠️ Le contenu amélioré semble invalide, tentative de réassemblage manuel...")
                
                # Récupérer le contenu amélioré pour chaque section
                improved_sections = {}
                for task in linking_tasks:
                    task_info = task_to_section_map.get(id(task))
                    if task_info and hasattr(task, 'output'):
                        section_id = task_info["section_id"]
                        # Copier la section originale et mettre à jour son contenu
                        improved_sections[section_id] = task_info["section_data"].copy()
                        # Extraire le contenu de la tâche, sans l'éventuel en-tête
                        task_content = MarkdownProcessor.extract_final_content(task.output)
                        improved_sections[section_id]['content'] = task_content
                
                # S'il y a des sections manquantes, utiliser les originales
                for section_id, section_data in sections.items():
                    if section_id not in improved_sections:
                        improved_sections[section_id] = section_data
                
                # Réassembler le document manuellement
                markdown_content_improved = MarkdownProcessor.reassemble(improved_sections)
            
            return markdown_content_improved.strip()
            
        except Exception as e:
            print(f"⚠️ Erreur lors de l'amélioration du contenu: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            
            # Tentative de récupération en gardant la structure originale
            try:
                print("🔄 Tentative de récupération des sections améliorées disponibles...")
                improved_sections = {}
                
                # Vérifier si des tâches ont été complétées avec succès
                for task in linking_tasks:
                    task_info = task_to_section_map.get(id(task))
                    if task_info and hasattr(task, 'output') and task.output:
                        section_id = task_info["section_id"]
                        # Copier la section originale et mettre à jour son contenu si possible
                        improved_sections[section_id] = task_info["section_data"].copy()
                        try:
                            # Extraire le contenu de la tâche, sans l'éventuel en-tête
                            task_content = MarkdownProcessor.extract_final_content(task.output)
                            if task_content and len(task_content) > 10:  # Vérifier que le contenu est valide
                                improved_sections[section_id]['content'] = task_content
                        except Exception:
                            # En cas d'erreur, garder le contenu original
                            pass
                
                # Ajouter les sections originales qui n'ont pas été traitées
                for section_id, section_data in sections.items():
                    if section_id not in improved_sections:
                        improved_sections[section_id] = section_data
                
                # Réassembler le document
                recovered_content = MarkdownProcessor.reassemble(improved_sections)
                if recovered_content and len(recovered_content) > len(markdown_content) / 2:
                    print("✅ Récupération réussie avec préservation de la structure originale")
                    return recovered_content.strip()
            except Exception as recovery_error:
                print(f"⚠️ Échec de la récupération: {recovery_error}")
            
            # En dernier recours, retourner le contenu original
            print("⚠️ Retour au contenu original en raison d'erreurs de traitement")
            return markdown_content