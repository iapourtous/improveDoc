"""
Traitement du contenu Markdown pour ImproveDoc.

Ce module contient les fonctions de traitement du contenu Markdown.
"""

import re
from typing import Dict, List, Tuple, Any

class MarkdownProcessor:
    """
    Classe pour le traitement du contenu Markdown.
    """
    
    @staticmethod
    def parse_sections(markdown_content: str) -> Dict[str, Dict[str, Any]]:
        """
        Divise le contenu Markdown en sections basées sur les en-têtes.
        Préserve la structure complète du document original.
        
        Args:
            markdown_content: Contenu Markdown à analyser
            
        Returns:
            Dictionnaire des sections avec leur contenu, incluant les niveaux de titre
        """
        # Ajout d'un marqueur de fin pour capturer la dernière section
        content_with_end_marker = markdown_content + "\n# __END__"
        
        # Trouver tous les en-têtes (de niveau 1, 2, 3, etc.)
        headers = re.finditer(r'(?m)^(#+)\s+(.*?)$', content_with_end_marker)
        
        # Stocker les positions et infos de tous les en-têtes
        header_positions = []
        for match in headers:
            header_positions.append({
                'start': match.start(),
                'end': match.end(),
                'level': len(match.group(1)),
                'text': match.group(2).strip(),
                'full_header': match.group(0)
            })
        
        # Initialiser le résultat
        sections = {}
        
        # Traiter le contenu avant le premier en-tête si présent
        if header_positions and header_positions[0]['start'] > 0:
            intro_content = content_with_end_marker[:header_positions[0]['start']].strip()
            if intro_content:
                sections['_intro_'] = {
                    'level': 0,
                    'title': '',
                    'content': intro_content,
                    'full_header': ''
                }
        
        # Traiter chaque section entre les en-têtes
        for i in range(len(header_positions) - 1):  # -1 pour exclure le marqueur __END__
            current = header_positions[i]
            next_header = header_positions[i+1]
            
            section_id = f"section_{i}"
            section_content = content_with_end_marker[current['end']:next_header['start']].strip()
            
            sections[section_id] = {
                'level': current['level'],
                'title': current['text'],
                'content': section_content,
                'full_header': current['full_header'],
                'original_position': i
            }
        
        # Retirer la section __END__ si elle est présente
        for key in list(sections.keys()):
            if '__END__' in sections[key].get('title', ''):
                del sections[key]
                
        return sections
    
    @staticmethod
    def reassemble(sections: Dict[str, Dict[str, Any]]) -> str:
        """
        Réassemble les sections en un document Markdown complet en préservant 
        l'ordre et la structure originale.
        
        Args:
            sections: Dictionnaire des sections avec leur contenu et métadonnées
            
        Returns:
            Document Markdown complet
        """
        # Trier les sections par position originale
        sorted_sections = []
        
        # D'abord l'intro si présente
        if '_intro_' in sections:
            sorted_sections.append({
                'header': '',
                'content': sections['_intro_']['content']
            })
        
        # Ensuite les autres sections triées par position originale
        section_items = [(k, v) for k, v in sections.items() if k != '_intro_']
        section_items.sort(key=lambda x: x[1].get('original_position', 999))
        
        for _, section in section_items:
            level = section['level']
            title = section['title']
            content = section['content']
            
            # Recréer l'en-tête avec le bon nombre de #
            header = '#' * level + ' ' + title
            
            sorted_sections.append({
                'header': header,
                'content': content
            })
        
        # Construire le document final
        result = []
        for section in sorted_sections:
            if section['header']:
                result.append(section['header'])
            if section['content']:
                result.append(section['content'])
            result.append('')  # Ligne vide entre les sections
        
        return '\n'.join(result).strip()
    
    @staticmethod
    def extract_final_content(result_text: str) -> str:
        """
        Extrait le contenu final d'une réponse qui peut contenir des blocs de code Markdown.
        
        Args:
            result_text: Texte de la réponse
            
        Returns:
            Contenu final extrait
        """
        # Chercher le contenu entre triples backticks s'il y en a
        code_blocks = re.findall(r'```(?:markdown)?\s*([\s\S]*?)\s*```', result_text)
        if code_blocks:
            # Prendre le bloc de code le plus long (probablement le contenu complet)
            final_content = max(code_blocks, key=len)
        else:
            # Sinon, utiliser tout le texte de la réponse
            final_content = result_text
        
        return final_content.strip()