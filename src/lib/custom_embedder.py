"""
Module contenant un embedder personnalisé pour ImproveDoc.

Ce module définit une classe d'embedder personnalisée basée sur SentenceTransformers.
"""
from typing import List
from chromadb.api.types import Documents, Embeddings
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer

class MultilingualE5Embedder(EmbeddingFunction):
    """
    Fonction d'embedding personnalisée utilisant le modèle multilingual-e5-large d'intfloat.
    
    Cette classe crée des embeddings pour les documents en utilisant le modèle
    'intfloat/multilingual-e5-large' de SentenceTransformer.
    """
    
    def __init__(self):
        """Initialise l'embedder avec le modèle multilingual-e5-large."""
        self.model = SentenceTransformer('intfloat/multilingual-e5-large')
    
    def __call__(self, input_texts: Documents) -> Embeddings:
        """
        Génère des embeddings pour les documents fournis.
        
        Args:
            input_texts: Liste de textes pour lesquels générer des embeddings
            
        Returns:
            Liste d'embeddings (vecteurs de représentation) pour les textes fournis
        """
        if not input_texts:
            return []
            
        # Préparer les textes avec le préfixe recommandé pour le modèle E5
        processed_texts = [f"passage: {text}" for text in input_texts]
        
        # Générer les embeddings avec le modèle
        embeddings = self.model.encode(processed_texts, convert_to_tensor=False)
        
        # Convertir en liste pour la compatibilité avec l'API ChromaDB
        return embeddings.tolist()