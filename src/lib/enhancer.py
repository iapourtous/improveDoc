"""
Service principal d'amélioration de contenu Markdown pour ImproveDoc.

Ce module contient le service principal d'amélioration de contenu Markdown.
"""

import traceback
from typing import Dict, List, Optional
import os

from crewai import Agent, Task, Crew, Process, LLM
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage

# Imports relatifs pour les modules du package
from .config import Config
from .agents import AgentFactory
from .content_processor import MarkdownProcessor
from .custom_embedder import MultilingualE5Embedder

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
        
        # Configuration du chemin de stockage pour la mémoire à partir du fichier .env
        storage_path = os.getenv("CREWAI_STORAGE_DIR", "./storage")
        os.makedirs(storage_path, exist_ok=True)
        print(f"Stockage de la mémoire configuré dans: {storage_path}")
        
        # Configuration de l'embedder personnalisé
        self.custom_embedder = MultilingualE5Embedder()
        
        # Configuration des mémoires
        self.long_term_memory = LongTermMemory(
            storage=LTMSQLiteStorage(
                db_path=f"{storage_path}/improve_doc_ltm.db"
            )
        )
        
        self.short_term_memory = ShortTermMemory(
            storage=RAGStorage(
                embedder_config={
                    "provider": "custom",
                    "config": {
                        "embedder": self.custom_embedder
                    }
                },
                type="short_term",
                path=f"{storage_path}/"
            )
        )
        
        self.entity_memory = EntityMemory(
            storage=RAGStorage(
                embedder_config={
                    "provider": "custom",
                    "config": {
                        "embedder": self.custom_embedder
                    }
                },
                type="entity",
                path=f"{storage_path}/"
            )
        )
        
        # Création de la factory d'agents avec la langue spécifiée
        self.agent_factory = AgentFactory(self.llm, wiki_language=self.config.wikipedia_language)
        
        print(f"ContentEnhancer configuré avec modèle: {self.config.model} et mémoire activée")
    
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
            description=f"""ENRICHIR SANS RACCOURCIR : Ajouter des informations Wikipédia à la section "{section_name}" tout en conservant INTÉGRALEMENT le contenu original.
            
            CONTENU ACTUEL DE LA SECTION À CONSERVER INTÉGRALEMENT:
            ```
            {section_content}
            ```
            
            MISSION PRINCIPALE:
            CONSERVER 100% du contenu original ET y ajouter des enrichissements Wikipédia pertinents.
            
            INSTRUCTIONS DÉTAILLÉES:
            1. CONSERVER INTÉGRALEMENT : Reproduis d'abord TOUT le contenu original tel quel, mot pour mot
            2. IDENTIFIER : Repère les concepts, termes techniques ou sujets qui mériteraient d'être approfondis
            3. RECHERCHER : Trouve des informations complémentaires sur Wikipédia
            4. ENRICHIR : Ajoute ces nouvelles informations EN PLUS du contenu original, en les intégrant naturellement
            5. CONTEXTUALISER : Lie les enrichissements au contenu existant avec des transitions fluides
            
            RÈGLES ABSOLUES - INTERDIT DE:
            - Supprimer ou raccourcir le moindre élément du contenu original
            - Remplacer des phrases existantes par des versions "améliorées"
            - Réorganiser ou restructurer le contenu original
            - Paraphraser ou reformuler les passages existants
            - Créer de nouvelles sous-sections qui n'existaient pas
            
            RÈGLES OBLIGATOIRES - TU DOIS:
            - Conserver chaque phrase, chaque liste, chaque détail du contenu original
            - Ajouter les enrichissements APRÈS ou ENTRE les éléments existants
            - Utiliser des formulations comme "De plus, selon Wikipédia...", "Il est également important de noter que...", "Pour approfondir ce point..."
            - Préserver absolument la structure Markdown existante
            - Maintenir le style et le ton du document original
            
            RÉSULTAT ATTENDU:
            Le contenu original complet + des ajouts d'informations Wikipédia = Un contenu plus long et plus riche, jamais plus court.
            
            Fournis le contenu de section ENRICHI (original + ajouts), sans titre de section.
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
            description=f"""VÉRIFIER SANS RÉDUIRE : Vérifier la véracité tout en conservant l'intégralité du contenu de la section "{section_name}" enrichie.
            
            CONTENU DE LA SECTION ENRICHIE À PRÉSERVER:
            ```
            {{{{enrichment_task.output}}}}
            ```
            
            MISSION PRINCIPALE:
            Vérifier les faits SANS supprimer ou raccourcir le contenu existant.
            
            INSTRUCTIONS:
            1. CONSERVER INTÉGRALEMENT : Reproduis tout le contenu reçu en entrée
            2. IDENTIFIER : Repère les affirmations factuelles importantes à vérifier
            3. VÉRIFIER : Contrôle chaque affirmation avec des sources fiables (Wikipédia)
            4. CORRIGER SI NÉCESSAIRE : Uniquement si une information est manifestement fausse
            5. NUANCER : Ajoute des précisions ou nuances si nécessaire, EN PLUS du contenu existant
            
            RÈGLES ABSOLUES - INTERDIT DE:
            - Supprimer des phrases, paragraphes ou informations
            - Raccourcir ou condenser le contenu
            - Remplacer des formulations par des versions "plus correctes"
            - Réorganiser la structure existante
            
            RÈGLES OBLIGATOIRES - TU DOIS:
            - Préserver chaque élément du contenu reçu
            - Ajouter les corrections ou nuances EN PLUS, pas EN REMPLACEMENT
            - Utiliser des formulations comme "Précision importante:", "Note de vérification:", "Selon les sources actuelles..."
            - Citer brièvement Wikipédia uniquement pour les corrections majeures
            - Maintenir le flux logique et le style du texte
            
            RÉSULTAT ATTENDU:
            Le contenu enrichi complet + éventuelles corrections/précisions = Jamais moins de contenu qu'en entrée.
            
            Fournis la section complète avec les vérifications effectuées.
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
            description=f"""AJOUTER LIENS SANS MODIFIER : Ajouter des liens Wikipédia dans la section "{section_name}" sans altérer le contenu.
            
            CONTENU DE LA SECTION VÉRIFIÉE À PRÉSERVER INTÉGRALEMENT:
            ```
            {{{{verification_task.output}}}}
            ```
            
            MISSION PRINCIPALE:
            Ajouter des liens Wikipédia SANS modifier, supprimer ou raccourcir le moindre contenu textuel.
            
            INSTRUCTIONS:
            1. CONSERVER INTÉGRALEMENT : Reproduis tout le contenu textuel reçu, mot pour mot
            2. IDENTIFIER : Repère les termes, concepts et entités qui méritent des liens
            3. RECHERCHER : Trouve les articles Wikipédia correspondants les plus pertinents
            4. LIER : Transforme uniquement les termes sélectionnés en liens Markdown [terme](https://fr.wikipedia.org/wiki/Page)
            5. VÉRIFIER : Assure-toi qu'aucun contenu textuel n'a été perdu ou modifié
            
            RÈGLES ABSOLUES - INTERDIT DE:
            - Modifier, supprimer ou raccourcir le moindre texte
            - Reformuler des phrases pour "améliorer" le style
            - Réorganiser la structure ou l'ordre du contenu
            - Changer la ponctuation ou la formulation
            - Supprimer des détails pour faire de la place aux liens
            
            RÈGLES OBLIGATOIRES - TU DOIS:
            - Conserver chaque mot, chaque phrase, chaque détail du contenu reçu
            - Ajouter uniquement des liens en transformant les termes existants
            - Éviter de lier des termes trop génériques ou évidents
            - Privilégier la première occurrence significative de chaque terme
            - Éviter les liens dans les titres et sous-titres
            - Limiter à 3-5 liens pertinents par section pour ne pas surcharger
            
            RÉSULTAT ATTENDU:
            Le contenu exact reçu en entrée + liens Wikipédia sur les termes appropriés = Même longueur de contenu avec liens enrichis.
            
            Fournis la section complète avec les liens ajoutés, sans aucune perte de contenu.
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
            description=f"""ASSEMBLER SANS PERDRE : Finaliser le document en préservant TOUT le contenu enrichi et la structure originale.
            
            STRUCTURE ORIGINALE DU DOCUMENT À RESPECTER STRICTEMENT:
            ```
            {original_structure}
            ```
            
            SECTIONS ENRICHIES AVEC LIENS À CONSERVER INTÉGRALEMENT:
            {chr(10).join(sections_description)}
            
            MISSION PRINCIPALE:
            Assembler un document final qui contient TOUT le contenu enrichi sans aucune perte.
            
            INSTRUCTIONS:
            1. CONSERVER INTÉGRALEMENT : Reproduis tout le contenu de chaque section enrichie, sans exception
            2. STRUCTURER : Respecte exactement la hiérarchie de titres du document original
            3. FORMATER : Améliore uniquement la présentation Markdown (sauts de ligne, espacement)
            4. HARMONISER : Assure une cohérence stylistique sans supprimer de contenu
            5. VÉRIFIER : Contrôle que tous les enrichissements et liens sont préservés
            
            RÈGLES ABSOLUES - INTERDIT DE:
            - Supprimer, raccourcir ou condenser le moindre contenu enrichi
            - Modifier les titres originaux (niveau ou texte)
            - Réorganiser l'ordre des sections
            - Éliminer des détails pour "améliorer" la lisibilité
            - Supprimer des liens Wikipédia ajoutés
            - Paraphraser ou reformuler le contenu existant
            
            RÈGLES OBLIGATOIRES - TU DOIS:
            - Inclure chaque phrase, chaque liste, chaque enrichissement reçu
            - Préserver tous les liens Wikipédia ajoutés par les agents précédents
            - Maintenir la structure exacte de titres du document original
            - Améliorer uniquement la présentation (espacement, formatage Markdown)
            - Assurer la cohérence du style de formatage
            - Créer un document PLUS LONG et PLUS RICHE que l'original
            
            RÉSULTAT ATTENDU:
            Un document Markdown complet = contenu original + tous les enrichissements + tous les liens + formatage amélioré.
            Le document final doit être significativement plus long et plus informatif que l'original.
            
            Fournis le document Markdown complet qui préserve TOUT le contenu enrichi.
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
        wiki_linker_agent = self.agent_factory.create_wiki_linker_agent()
        markdown_editor_agent = self.agent_factory.create_markdown_editor_agent()
        
        # 3. Générer dynamiquement les tâches pour chaque section
        enrichment_tasks = []
        linking_tasks = []
        
        # Ordonner les sections par leur position originale dans le document
        section_items = sorted(
            [(k, v) for k, v in sections.items()], 
            key=lambda x: x[1].get('original_position', 999)
        )
        
        # Traiter toutes les sections sans limitation pour préserver l'intégralité du document
        # (Ancienne limitation supprimée pour éviter la troncature du contenu)
        
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
            
            # Tâche d'ajout de liens (dépend de l'enrichissement)
            linking_task = self._create_linking_task(
                wiki_linker_agent, section_name, enrichment_task
            )
            linking_tasks.append(linking_task)
            
            # Stocker la référence à la section d'origine dans un dictionnaire séparé
            task_to_section_map[id(linking_task)] = {
                "section_id": section_id,
                "section_data": section_data
            }
        
        # 4. Tâche d'édition Markdown globale
        markdown_editing_task = self._create_editing_task(
            markdown_editor_agent, linking_tasks, original_structure
        )
        
        # 5. Création et exécution de l'équipage avec mémoire
        crew = Crew(
            agents=[research_agent, wiki_linker_agent, markdown_editor_agent],
            tasks=enrichment_tasks + linking_tasks + [markdown_editing_task],
            verbose=True,
            process=Process.sequential,
            memory=True,
            long_term_memory=self.long_term_memory,
            short_term_memory=self.short_term_memory,
            entity_memory=self.entity_memory
        )
        
        try:
            result = crew.kickoff()
            
            # Dans les nouvelles versions de CrewAI, crew.kickoff() retourne un objet CrewOutput
            # au lieu d'une chaîne de caractères, donc nous devons extraire le texte
            result_text = str(result) if hasattr(result, '__str__') else ""
            
            # Extraire le contenu final de la réponse
            markdown_content_improved = MarkdownProcessor.extract_final_content(result_text)

            # ------------------------------------------------------------------
            # Post‑traitement : éviter les sections vides ou placeholders
            # ------------------------------------------------------------------
            try:
                improved_sections_map = MarkdownProcessor.parse_sections(markdown_content_improved)
                updated = False
                for sec_id, sec in improved_sections_map.items():
                    content = sec.get("content", "").strip()
                    orig_content = sections.get(sec_id, {}).get("content", "").strip()

                    # Considérer la section « vide » si aucune phrase ou uniquement un placeholder HTML
                    is_placeholder = content.startswith("<!--") or len(content) < 20
                    if is_placeholder and orig_content:
                        improved_sections_map[sec_id]["content"] = orig_content
                        updated = True

                if updated:
                    markdown_content_improved = MarkdownProcessor.reassemble(improved_sections_map)
            except Exception as _:
                # Si le post‑traitement échoue, on garde la version telle quelle.
                pass
            
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