#!/usr/bin/env python3
"""
ImproveDoc - Outil d'amélioration de contenu Markdown avec Wikipedia et CrewAI

Cet outil prend en entrée un fichier Markdown et produit une version améliorée 
en enrichissant le contenu avec des informations provenant de Wikipedia.
"""

import os
import re
import sys
import argparse
from typing import Dict, List, Optional
from pathlib import Path

# Désactiver la télémétrie avant d'importer crewai
os.environ["CREWAI_TELEMETRY"] = "False"
os.environ["TELEMETRY_ENABLED"] = "False"
os.environ["OPENTELEMETRY_ENABLED"] = "False"

# Désactiver les warnings
import warnings
warnings.filterwarnings("ignore", category=Warning)

try:
    from crewai import Agent, Task, Crew, Process, LLM
    from dotenv import load_dotenv
except ImportError:
    print("⚠️ Certaines dépendances requises ne sont pas installées.")
    print("Installez-les avec : pip install -r requirements.txt")
    sys.exit(1)

# Supprimer les modules de télémétrie s'ils sont chargés
for mod in list(sys.modules.keys()):
    if 'telemetry' in mod or 'opentelemetry' in mod:
        if mod in sys.modules:
            del sys.modules[mod]

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

class WikipediaTool:
    """
    Outil pour rechercher des informations sur Wikipedia.
    """
    
    def __init__(self, language: str = "fr"):
        """
        Initialise l'outil Wikipedia.
        
        Args:
            language: Langue pour les recherches Wikipedia ("fr", "en", etc.)
        """
        self.language = language
        
        try:
            import wikipedia
            # Configurer la langue de Wikipedia
            wikipedia.set_lang(language)
            self.wikipedia = wikipedia
        except ImportError:
            print("⚠️ Le module 'wikipedia' n'est pas installé.")
            print("Installez-le avec : pip install wikipedia")
            sys.exit(1)
    
    def search(self, query: str, results: int = 5) -> List[str]:
        """
        Recherche des pages Wikipedia correspondant à une requête.
        
        Args:
            query: Terme de recherche
            results: Nombre de résultats à retourner
            
        Returns:
            Liste des titres de pages correspondants
        """
        try:
            return self.wikipedia.search(query, results=results)
        except Exception as e:
            print(f"⚠️ Erreur lors de la recherche Wikipedia: {e}")
            return []
    
    def get_summary(self, page_title: str, sentences: int = 5) -> str:
        """
        Récupère le résumé d'une page Wikipedia.
        
        Args:
            page_title: Titre de la page
            sentences: Nombre de phrases à récupérer
            
        Returns:
            Résumé de la page
        """
        try:
            return self.wikipedia.summary(page_title, sentences=sentences)
        except Exception as e:
            print(f"⚠️ Erreur lors de la récupération du résumé de '{page_title}': {e}")
            return ""
    
    def get_content(self, page_title: str) -> str:
        """
        Récupère le contenu complet d'une page Wikipedia.
        
        Args:
            page_title: Titre de la page
            
        Returns:
            Contenu complet de la page
        """
        try:
            page = self.wikipedia.page(page_title)
            return page.content
        except Exception as e:
            print(f"⚠️ Erreur lors de la récupération du contenu de '{page_title}': {e}")
            return ""
    
    def get_url(self, page_title: str) -> str:
        """
        Récupère l'URL d'une page Wikipedia.
        
        Args:
            page_title: Titre de la page
            
        Returns:
            URL de la page
        """
        try:
            page = self.wikipedia.page(page_title)
            return page.url
        except Exception as e:
            print(f"⚠️ Erreur lors de la récupération de l'URL de '{page_title}': {e}")
            return ""

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
        # Configuration des paramètres LLM
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model or os.getenv('LLM_MODEL', 'gpt-3.5-turbo')
        
        # Valeurs par défaut pour les autres paramètres
        self.temperature = float(os.getenv('LLM_TEMPERATURE', '0.7'))
        self.max_tokens = int(os.getenv('LLM_MAX_TOKENS', '4096'))
        
        # Vérifier que la clé API est configurée
        if not self.api_key:
            print("⚠️ Aucune clé API n'a été configurée.")
            print("Définissez la variable d'environnement OPENAI_API_KEY ou utilisez l'option --api-key.")
            sys.exit(1)
        
        # Création de l'outil Wikipedia
        self.wikipedia_tool = WikipediaTool(language="fr")
        
        # Configuration du LLM pour CrewAI
        self.llm = LLM(
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        print(f"ContentEnhancer configuré avec modèle: {self.model}")
    
    def _parse_markdown_sections(self, markdown_content: str) -> Dict[str, str]:
        """
        Divise le contenu Markdown en sections basées sur les en-têtes.
        
        Args:
            markdown_content: Contenu Markdown à analyser
            
        Returns:
            Dictionnaire des sections avec leur contenu
        """
        # Ajout d'un marqueur de fin pour capturer la dernière section
        content_with_end_marker = markdown_content + "\n## __END__"
        
        # Trouver tous les en-têtes de niveau 2 (##)
        headers = re.finditer(r'(?m)^##\s+(.*?)$', content_with_end_marker)
        
        sections = {}
        last_pos = 0
        last_header = "Introduction"  # Section par défaut pour le contenu avant le premier ##
        
        for match in headers:
            header_start = match.start()
            header_text = match.group(1).strip()
            
            # Si ce n'est pas le premier en-tête, ajouter le contenu de la section précédente
            if last_pos > 0:
                section_content = content_with_end_marker[last_pos:header_start].strip()
                sections[last_header] = section_content
            elif header_start > 0:
                # Contenu avant le premier en-tête
                intro_content = content_with_end_marker[:header_start].strip()
                if intro_content:
                    sections["Introduction"] = intro_content
            
            # Mettre à jour pour la prochaine itération
            last_pos = match.end()
            last_header = header_text
        
        # Supprimer la section __END__ si elle a été ajoutée
        if "__END__" in sections:
            del sections["__END__"]
            
        return sections
    
    def _reassemble_markdown(self, sections: Dict[str, str]) -> str:
        """
        Réassemble les sections en un document Markdown complet.
        
        Args:
            sections: Dictionnaire des sections avec leur contenu
            
        Returns:
            Document Markdown complet
        """
        # Ordonner les sections avec Introduction en premier si présente
        ordered_sections = []
        if "Introduction" in sections:
            intro_content = sections["Introduction"]
            if intro_content:
                ordered_sections.append(intro_content)  # Pas d'en-tête pour l'intro
            del sections["Introduction"]
        
        # Ajouter les autres sections
        for header, content in sections.items():
            if header != "__END__":
                section_text = f"## {header}\n\n{content}\n"
                ordered_sections.append(section_text)
        
        # Joindre toutes les sections avec des sauts de ligne
        return "\n".join(ordered_sections)
    
    def _create_research_agent(self) -> Agent:
        """
        Crée un agent de recherche pour enrichir le contenu avec des informations de Wikipédia.
        
        Returns:
            Agent CrewAI pour la recherche et l'enrichissement
        """
        return Agent(
            role="Chercheur Wikipédia",
            goal="Enrichir le contenu avec des informations pertinentes et factuelles de Wikipédia",
            backstory="""Tu es un expert en recherche documentaire capable d'extraire 
            des informations précises et pertinentes de Wikipédia pour enrichir un contenu existant.
            Tu sais identifier les lacunes informatives dans un texte et les combler avec
            des données fiables. Tu maîtrises l'art d'intégrer harmonieusement de nouvelles 
            informations tout en préservant la structure et le style du document original.""",
            tools=[
                self.wikipedia_tool.search,
                self.wikipedia_tool.get_summary,
                self.wikipedia_tool.get_content,
                self.wikipedia_tool.get_url
            ],
            allow_delegation=False,
            verbose=True,
            llm=self.llm
        )
    
    def _create_fact_checker_agent(self) -> Agent:
        """
        Crée un agent de vérification des faits.
        
        Returns:
            Agent CrewAI pour la vérification factuelle
        """
        return Agent(
            role="Vérificateur de Faits",
            goal="Vérifier la véracité des affirmations présentes dans le contenu",
            backstory="""Tu es un fact-checker méticuleux qui vérifie scrupuleusement 
            chaque affirmation importante. Tu compares les informations avec des sources
            fiables comme Wikipédia pour confirmer leur exactitude. Tu sais détecter les
            inexactitudes, les simplifications excessives et les affirmations non étayées.
            Tu proposes des corrections précises basées uniquement sur des sources vérifiables.""",
            tools=[
                self.wikipedia_tool.search,
                self.wikipedia_tool.get_summary,
                self.wikipedia_tool.get_content,
                self.wikipedia_tool.get_url
            ],
            allow_delegation=False,
            verbose=True,
            llm=self.llm
        )
    
    def _create_wiki_linker_agent(self) -> Agent:
        """
        Crée un agent pour l'ajout de liens Wikipédia pertinents.
        
        Returns:
            Agent CrewAI pour l'ajout de liens
        """
        return Agent(
            role="Spécialiste des Liens Wikipédia",
            goal="Identifier et lier les termes pertinents vers des articles Wikipédia",
            backstory="""Tu es un expert en hypertexte et en organisation de contenu. 
            Tu excelles dans l'identification des termes, concepts et entités qui méritent
            d'être liés à leurs articles Wikipédia correspondants. Tu sais distinguer les
            termes vraiment importants des termes secondaires, et tu places les liens de
            manière stratégique pour enrichir la navigation sans surcharger le texte.""",
            tools=[
                self.wikipedia_tool.search,
                self.wikipedia_tool.get_url
            ],
            allow_delegation=False,
            verbose=True,
            llm=self.llm
        )
    
    def _create_markdown_editor_agent(self) -> Agent:
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
    
    def enhance_content(self, markdown_content: str) -> str:
        """
        Améliore un contenu Markdown avec des informations Wikipédia, vérification factuelle,
        et ajout de liens pertinents.
        
        Args:
            markdown_content: Contenu Markdown original à améliorer
            
        Returns:
            Contenu Markdown amélioré
        """
        # Afficher un message pour débug
        print(f"ContentEnhancer.enhance_content: Démarrage de l'amélioration du contenu")
        print(f"Longueur du contenu: {len(markdown_content)} caractères")
        
        # 1. Diviser le contenu en sections
        sections = self._parse_markdown_sections(markdown_content)
        
        # Afficher les sections trouvées
        print(f"Sections trouvées: {list(sections.keys())}")
        
        # 2. Créer les agents
        research_agent = self._create_research_agent()
        fact_checker_agent = self._create_fact_checker_agent()
        wiki_linker_agent = self._create_wiki_linker_agent()
        markdown_editor_agent = self._create_markdown_editor_agent()
        
        # 3. Générer dynamiquement les tâches pour chaque section
        enrichment_tasks = []
        verification_tasks = []
        linking_tasks = []
        
        # Limiter le nombre de sections traitées si nécessaire
        section_keys = list(sections.keys())[:5]  # Limiter à 5 sections pour la performance
        
        for section_name in section_keys:
            section_content = sections[section_name]
            
            # Tâche d'enrichissement pour cette section
            enrichment_task = Task(
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
                
                CONTRAINTES:
                - N'ajoute que des informations factuelles et vérifiables
                - Maintiens la cohérence avec le reste du document
                - Préserve tous les points importants du contenu original
                - Respecte la structure Markdown
                
                Fournis la section complète enrichie, pas seulement les ajouts.
                """,
                agent=research_agent,
                expected_output=f"Section '{section_name}' enrichie avec des informations de Wikipédia"
            )
            enrichment_tasks.append(enrichment_task)
            
            # Tâche de vérification factuelle (dépend de l'enrichissement)
            verification_task = Task(
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
                agent=fact_checker_agent,
                context=[enrichment_task],
                expected_output=f"Section '{section_name}' avec affirmations vérifiées"
            )
            verification_tasks.append(verification_task)
            
            # Tâche d'ajout de liens (dépend de la vérification)
            linking_task = Task(
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
                agent=wiki_linker_agent,
                context=[verification_task],
                expected_output=f"Section '{section_name}' avec liens Wikipédia ajoutés"
            )
            linking_tasks.append(linking_task)
        
        # 4. Tâche d'édition Markdown globale (après que toutes les sections aient été traitées)
        markdown_editing_task = Task(
            description=f"""Assure la qualité et la cohérence du format Markdown du document entier.
            
            SECTIONS AMÉLIORÉES AVEC LIENS:
            {", ".join([f"{{{{linking_tasks[{i}].output}}}}" for i in range(len(linking_tasks))])}
            
            INSTRUCTIONS:
            1. Assure-toi que les titres sont correctement hiérarchisés
            2. Vérifie que les listes sont bien formatées
            3. Ajoute des sauts de ligne appropriés pour améliorer la lisibilité
            4. Harmonise le style de formatage dans tout le document
            
            CONTRAINTES:
            - Préserve tout le contenu substantiel
            - Conserve les liens Wikipédia ajoutés
            - Ne modifie pas les informations factuelles importantes
            
            Fournis le document Markdown complet correctement formaté.
            """,
            agent=markdown_editor_agent,
            context=linking_tasks,
            expected_output="Document Markdown avec format amélioré"
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
            # Chercher le contenu entre triples backticks s'il y en a
            code_blocks = re.findall(r'```(?:markdown)?\s*([\s\S]*?)\s*```', result_text)
            if code_blocks:
                # Prendre le bloc de code le plus long (probablement le contenu complet)
                final_content = max(code_blocks, key=len)
            else:
                # Sinon, utiliser tout le texte de la réponse
                final_content = result_text
            
            return final_content.strip()
            
        except Exception as e:
            import traceback
            print(f"⚠️ Erreur lors de l'amélioration du contenu: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            # En cas d'erreur, retourner le contenu original
            return markdown_content

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
    
    # Créer le service d'amélioration de contenu
    enhancer = ContentEnhancer(api_key=args.api_key, model=args.model)
    
    # Améliorer le contenu
    print(f"🔍 Analyse et amélioration du contenu en cours...")
    improved_content = enhancer.enhance_content(content)
    
    # Écrire le contenu amélioré dans le fichier de sortie
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(improved_content)
    except Exception as e:
        print(f"⚠️ Erreur lors de l'écriture du fichier: {e}")
        sys.exit(1)
    
    print(f"✅ Contenu amélioré enregistré dans {output_path}")

if __name__ == "__main__":
    main()