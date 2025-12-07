"""
app.py - Interface Gradio pour Hugging Face Spaces
Moteur de Recherche Sémantique avec IA
"""

import gradio as gr
from indexer_optimized import get_indexer
import time
import os

print("=" * 60)
print("🔄 Initialisation du moteur de recherche...")
print("=" * 60)

try:
    indexer = get_indexer()
    total_pages = len(indexer.pages_data)
    print(f"✓ Moteur prêt ! {total_pages} pages indexées")
except Exception as e:
    print(f"⚠️ Erreur : {e}")
    indexer = None
    total_pages = 0

print("=" * 60)

def search_function(query, top_k, threshold):
    """Fonction de recherche principale"""
    if not indexer or not indexer.pages_data:
        return """
        <div style='text-align: center; padding: 40px; background: #fff3cd; border-radius: 10px;'>
            <h2>⚠️ Base de données vide</h2>
            <p>Aucune page indexée pour le moment.</p>
            <p>La base de données emb.db est peut-être manquante ou vide.</p>
        </div>
        """
    
    if not query or len(query.strip()) < 2:
        return """
        <div style='text-align: center; padding: 40px;'>
            <h2>🔍 Entrez votre recherche</h2>
            <p>Tapez au moins 2 caractères pour commencer</p>
        </div>
        """
    
    start = time.time()
    
    try:
        results = indexer.search(
            query_text=query.strip(), 
            top_k=int(top_k), 
            threshold=float(threshold)
        )
        elapsed = time.time() - start
        
        if not results:
            return f"""
            <div style='text-align: center; padding: 40px;'>
                <h2>🔍 Aucun résultat</h2>
                <p>Aucune page trouvée pour "<strong>{query}</strong>"</p>
                <p style='color: #666;'>Essayez avec des termes différents ou réduisez le seuil de pertinence</p>
            </div>
            """
        
        # HTML des résultats
        html = f"""
        <div style='padding: 20px;'>
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;'>
                <h2 style='margin: 0;'>🔍 Résultats pour "{query}"</h2>
                <p style='margin: 10px 0 0 0; opacity: 0.9;'>
                    {len(results)} résultats trouvés en {elapsed:.2f}s
                </p>
            </div>
        """
        
        for i, result in enumerate(results, 1):
            similarity_percent = result['similarity'] * 100
            
            if similarity_percent >= 70:
                badge_color = "#4caf50"
                badge_text = "Très pertinent"
            elif similarity_percent >= 50:
                badge_color = "#ff9800"
                badge_text = "Pertinent"
            else:
                badge_color = "#2196f3"
                badge_text = "Moyennement pertinent"
            
            html += f"""
            <div style='background: white; border-radius: 10px; padding: 20px; 
                        margin-bottom: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        border-left: 4px solid {badge_color};'>
                
                <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                    <span style='background: {badge_color}; color: white; 
                                 padding: 5px 12px; border-radius: 20px; 
                                 font-size: 0.85em; font-weight: 600;'>
                        {similarity_percent:.1f}% - {badge_text}
                    </span>
                </div>
                
                <h3 style='margin: 10px 0; color: #333; font-size: 1.2em;'>
                    {i}. {result['title'] or 'Sans titre'}
                </h3>
                
                <a href='{result['url']}' target='_blank' 
                   style='color: #667eea; text-decoration: none; 
                          font-size: 0.9em; word-break: break-all; display: inline-block; margin: 10px 0;'>
                    🔗 {result['url']}
                </a>
                
                <p style='margin: 15px 0 0 0; color: #666; line-height: 1.6;'>
                    {result['description'][:300] if result['description'] else 'Pas de description disponible'}
                    {'...' if len(result.get('description', '')) > 300 else ''}
                </p>
            </div>
            """
        
        html += "</div>"
        return html
        
    except Exception as e:
        return f"""
        <div style='text-align: center; padding: 40px; background: #f8d7da; border-radius: 10px;'>
            <h2>❌ Erreur</h2>
            <p>Une erreur s'est produite lors de la recherche :</p>
            <code>{str(e)}</code>
        </div>
        """

def get_stats():
    """Afficher les statistiques"""
    if not indexer or not indexer.pages_data:
        return """
        <div style='text-align: center; padding: 20px; background: #fff3cd; border-radius: 10px;'>
            <h3>⚠️ Aucune donnée</h3>
            <p>La base de données est vide</p>
        </div>
        """
    
    return f"""
    <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border-radius: 10px;'>
        <h3 style='color: #333; margin-bottom: 15px;'>📊 Statistiques du Moteur de Recherche</h3>
        <div style='font-size: 3em; font-weight: bold; color: #667eea; margin: 20px 0;'>
            {len(indexer.pages_data)}
        </div>
        <p style='font-size: 1.1em; color: #666;'>pages indexées et prêtes à être recherchées</p>
        <p style='margin-top: 20px; color: #999; font-size: 0.9em;'>
            Modèle : SentenceTransformers (all-MiniLM-L6-v2)
        </p>
    </div>
    """

# Interface Gradio
with gr.Blocks(theme=gr.themes.Soft(), title="🔍 Moteur de Recherche IA") as demo:
    
    gr.Markdown("""
    # 🔍 Moteur de Recherche Sémantique avec IA
    
    Recherche intelligente basée sur les **embeddings** et la **similarité sémantique**.
    
    Propulsé par **SentenceTransformers** pour une compréhension contextuelle de vos requêtes.
    """)
    
    with gr.Tab("🔍 Recherche"):
        with gr.Row():
            with gr.Column(scale=4):
                query_input = gr.Textbox(
                    label="Votre recherche",
                    placeholder="Ex: machine learning, intelligence artificielle, programmation Python...",
                    lines=1
                )
            with gr.Column(scale=1):
                search_btn = gr.Button("🚀 Rechercher", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                top_k_slider = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=10,
                    step=1,
                    label="📊 Nombre de résultats",
                    info="Combien de résultats afficher (5-50)"
                )
            with gr.Column():
                threshold_slider = gr.Slider(
                    minimum=0.0,
                    maximum=0.9,
                    value=0.25,
                    step=0.05,
                    label="🎯 Seuil de pertinence",
                    info="Plus élevé = résultats plus stricts (0.0-0.9)"
                )
        
        results_output = gr.HTML(label="Résultats")
        
        # Exemples
        gr.Examples(
            examples=[
                ["machine learning", 10, 0.25],
                ["python programming", 10, 0.30],
                ["artificial intelligence", 15, 0.25],
                ["web development", 10, 0.30],
                ["data science", 10, 0.25],
                ["neural networks", 10, 0.30],
            ],
            inputs=[query_input, top_k_slider, threshold_slider],
            label="💡 Exemples de recherche - Cliquez pour essayer"
        )
        
        # Actions
        search_btn.click(
            fn=search_function,
            inputs=[query_input, top_k_slider, threshold_slider],
            outputs=results_output
        )
        
        query_input.submit(
            fn=search_function,
            inputs=[query_input, top_k_slider, threshold_slider],
            outputs=results_output
        )
    
    with gr.Tab("📊 Statistiques"):
        stats_output = gr.HTML(value=get_stats())
        refresh_btn = gr.Button("🔄 Rafraîchir les statistiques")
        refresh_btn.click(fn=get_stats, outputs=stats_output)
    
    with gr.Tab("ℹ️ À propos"):
        gr.Markdown("""
        ## 🎯 Comment ça marche ?
        
        Ce moteur de recherche utilise l'intelligence artificielle pour comprendre le **sens** de vos requêtes, 
        pas seulement les mots-clés.
        
        ### 🔬 Technologies utilisées
        
        1. **SentenceTransformers** (`all-MiniLM-L6-v2`)
           - Convertit le texte en vecteurs numériques (embeddings)
           - Capture le sens contextuel des phrases
        
        2. **Similarité cosinus**
           - Compare les embeddings pour trouver les pages pertinentes
           - Score de 0 (pas pertinent) à 1 (très pertinent)
        
        3. **Indexation en mémoire**
           - Recherches ultra-rapides (< 1 seconde)
           - Base de données SQLite optimisée
        
        ### ✨ Fonctionnalités
        
        - ✅ **Recherche sémantique** : Comprend les synonymes et le contexte
        - ✅ **Résultats triés** : Par pertinence décroissante
        - ✅ **Réglages personnalisables** : Nombre de résultats et seuil
        - ✅ **Interface intuitive** : Design moderne avec Gradio
        - ✅ **Rapide** : Résultats en moins d'une seconde
        
        ### 📚 Base de données
        
        Contient des pages crawlées depuis divers sites web.
        
        ### 👨‍💻 Développement
        
        - **Framework** : Gradio + Python
        - **Hébergement** : Hugging Face Spaces / Local
        - **Modèle** : SentenceTransformers (open source)
        
        ---
        
        💡 **Astuce** : Pour de meilleurs résultats, utilisez des phrases complètes plutôt que des mots isolés !
        """)

# Lancer l'application
if __name__ == "__main__":
    demo.launch()