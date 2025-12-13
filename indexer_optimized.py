import sqlite3
import json
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Tuple
import time

DB_FILE = "emb.db"

class SimpleIndexer:
    def __init__(self):
        print("📄 Chargement des modèles ML/DL...")
        start = time.time()
        
        # Modèle principal (bi-encoder)
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Cross-encoder pour re-ranking
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # TF-IDF pour query expansion
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        print(f"✓ Modèles chargés en {time.time()-start:.1f}s")
        
        self.pages_data = []
        self.tfidf_fitted = False
        
        self.init_db()
        self.load_data()
    
    def init_db(self):
        """Initialisation DB"""
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                title TEXT,
                description TEXT,
                image_links TEXT,
                title_emb BLOB,
                desc_emb BLOB,
                combined_text TEXT,
                page_rank REAL DEFAULT 0.0
            )
        """)
        conn.commit()
        conn.close()
    
    def process_batch(self, batch):
        """Process batch avec features ML"""
        if not batch:
            return 0
        
        urls, titles, descs, images_lists = zip(*batch)
        
        print(f"📄 Encoding {len(batch)} pages avec ML...")
        
        # Texte combiné
        combined_texts = [f"{t} {d}" for t, d in zip(titles, descs)]
        
        # Embeddings
        title_embs = self.text_model.encode(list(titles), show_progress_bar=False)
        desc_embs = self.text_model.encode(list(descs), show_progress_bar=False)
        
        # Page rank simple
        page_ranks = [
            min(1.0, len(imgs) * 0.1 + len(desc) / 1000) 
            for imgs, desc in zip(images_lists, descs)
        ]
        
        # Save
        rows = []
        for i in range(len(batch)):
            rows.append((
                urls[i], 
                titles[i], 
                descs[i], 
                json.dumps(images_lists[i]),
                title_embs[i].astype(np.float32).tobytes(),
                desc_embs[i].astype(np.float32).tobytes(),
                combined_texts[i],
                float(page_ranks[i])
            ))
        
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.executemany("""
            INSERT OR REPLACE INTO pages
            (url, title, description, image_links, title_emb, desc_emb, combined_text, page_rank)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        conn.commit()
        conn.close()
        
        print(f"✓ Saved {len(batch)} pages")
        return len(batch)
    
    def load_data(self):
        """Charger données + fit TF-IDF"""
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        # Vérifier colonnes
        c.execute("PRAGMA table_info(pages)")
        columns = [col[1] for col in c.fetchall()]
        
        has_combined_text = 'combined_text' in columns
        has_page_rank = 'page_rank' in columns
        
        if has_combined_text and has_page_rank:
            query = """
                SELECT url, title, description, image_links, 
                       title_emb, desc_emb, combined_text, page_rank 
                FROM pages
            """
        else:
            query = """
                SELECT url, title, description, image_links, 
                       title_emb, desc_emb
                FROM pages
            """
        
        c.execute(query)
        rows = c.fetchall()
        conn.close()
        
        if not rows:
            print("⚠️  Aucune donnée indexée")
            self.pages_data = []
            return
        
        self.pages_data = []
        combined_texts = []
        
        for row in rows:
            if has_combined_text and has_page_rank:
                url, title, desc, img_links, title_emb_blob, desc_emb_blob, combined_text, page_rank = row
            else:
                url, title, desc, img_links, title_emb_blob, desc_emb_blob = row
                combined_text = None
                page_rank = 0.0
            
            title_emb = np.frombuffer(title_emb_blob, dtype=np.float32)
            desc_emb = np.frombuffer(desc_emb_blob, dtype=np.float32)
            
            if not combined_text:
                combined_text = f"{title} {desc}"
            
            self.pages_data.append({
                'url': url,
                'title': title,
                'description': desc,
                'image_links': json.loads(img_links) if img_links else [],
                'title_emb': title_emb,
                'desc_emb': desc_emb,
                'combined_text': combined_text,
                'page_rank': page_rank
            })
            combined_texts.append(combined_text)
        
        # Fit TF-IDF
        if combined_texts:
            print("📄 Fitting TF-IDF...")
            self.tfidf.fit(combined_texts)
            self.tfidf_matrix = self.tfidf.transform(combined_texts)
            self.tfidf_fitted = True
        
        print(f"✓ Loaded {len(self.pages_data)} pages + TF-IDF fitted")
    
    def _expand_query(self, query: str) -> str:
        """Query expansion avec TF-IDF"""
        if not self.tfidf_fitted:
            return query
        
        query_vec = self.tfidf.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        top_indices = similarities.argsort()[-3:][::-1]
        
        expanded_terms = set(query.lower().split())
        for idx in top_indices:
            if similarities[idx] > 0.1:
                text = self.pages_data[idx]['combined_text'].lower().split()
                expanded_terms.update(text[:5])
        
        return ' '.join(list(expanded_terms)[:20])
    
    def _semantic_search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """Recherche sémantique"""
        query_emb = self.text_model.encode([query], show_progress_bar=False)[0]
        
        results = []
        for idx, page in enumerate(self.pages_data):
            title_sim = cosine_similarity(
                query_emb.reshape(1, -1), 
                page['title_emb'].reshape(1, -1)
            )[0][0]
            
            desc_sim = cosine_similarity(
                query_emb.reshape(1, -1), 
                page['desc_emb'].reshape(1, -1)
            )[0][0]
            
            # Score combiné avec boost de 20% du page_rank
            similarity = max(title_sim, desc_sim) * (1 + page['page_rank'] * 0.2)
            results.append((idx, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _bm25_search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """BM25 search"""
        if not self.tfidf_fitted:
            return []
        
        query_vec = self.tfidf.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [(idx, similarities[idx]) for idx in top_indices]
    
    def _rerank(self, query: str, candidates: List[int], top_k: int) -> List[Dict]:
        """Re-ranking avec Cross-Encoder"""
        if not candidates:
            return []
        
        pairs = []
        for idx in candidates:
            page = self.pages_data[idx]
            doc_text = f"{page['title']} {page['description'][:200]}"
            pairs.append([query, doc_text])
        
        scores = self.reranker.predict(pairs)
        
        results = []
        for idx, score in zip(candidates, scores):
            page = self.pages_data[idx]
            results.append({
                'url': page['url'],
                'title': page['title'],
                'description': page['description'],
                'image_links': page['image_links'],
                'similarity': float(score),
                'page_rank': page['page_rank']
            })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def search(self, query_text: str, top_k: int = 20, threshold: float = 0.15, 
               use_reranking: bool = True, use_expansion: bool = True) -> List[Dict]:
        """Recherche hybride ML/DL"""
        if not query_text or not self.pages_data:
            return []
        
        start = time.time()
        
        # Query expansion
        expanded_query = self._expand_query(query_text) if use_expansion else query_text
        
        # Semantic search
        semantic_results = self._semantic_search(expanded_query, top_k=50)
        
        # BM25 search
        bm25_results = self._bm25_search(query_text, top_k=50)
        
        # Fusion (Reciprocal Rank Fusion)
        fused_scores = {}
        
        for rank, (idx, score) in enumerate(semantic_results, 1):
            fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (rank + 60)
        
        for rank, (idx, score) in enumerate(bm25_results, 1):
            fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (rank + 60)
        
        # Top candidats fusionnés
        top_candidates = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_indices = [idx for idx, _ in top_candidates[:50]]
        
        # Re-ranking
        if use_reranking and candidate_indices:
            results = self._rerank(query_text, candidate_indices, top_k)
        else:
            # Sans re-ranking: utiliser les scores sémantiques originaux
            results = []
            # Créer un dict des scores sémantiques originaux
            semantic_score_dict = {idx: score for idx, score in semantic_results}
            
            for idx in candidate_indices[:top_k]:
                page = self.pages_data[idx]
                # Utiliser le score sémantique original au lieu du score RRF
                original_score = semantic_score_dict.get(idx, 0.0)
                results.append({
                    'url': page['url'],
                    'title': page['title'],
                    'description': page['description'],
                    'image_links': page['image_links'],
                    'similarity': float(original_score),
                    'page_rank': page['page_rank']
                })
        
        # Filter par threshold
        results = [r for r in results if r['similarity'] > threshold]
        
        print(f"⚡ Recherche ML en {time.time()-start:.3f}s - {len(results)} résultats")
        
        return results

# Global instance
_indexer = None

def get_indexer():
    global _indexer
    if _indexer is None:
        _indexer = SimpleIndexer()
    return _indexer

def submit_batch_to_indexer(batch):
    indexer = get_indexer()
    count = indexer.process_batch(batch)
    indexer.load_data()
    return count