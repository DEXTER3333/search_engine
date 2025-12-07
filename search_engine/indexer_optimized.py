import sqlite3
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DB_FILE = "emb.db"

class SimpleIndexer:
    def __init__(self):
        print("🔄 Loading text model...")
        
        # Charger uniquement le modèle texte (pas d'images)
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.device = "cpu"
        
        # Pas de modèle image
        self.clip_model = None
        
        print("✓ Text model loaded!")
        
        self.pages_data = []
        self.init_db()
        self.load_data()
    
    def init_db(self):
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
                desc_emb BLOB
            )
        """)
        conn.commit()
        conn.close()
    
    def process_batch(self, batch):
        """Process a batch of pages"""
        if not batch:
            return 0
        
        urls, titles, descs, images_lists = zip(*batch)
        
        print(f"🔄 Encoding {len(batch)} pages...")
        
        # Encode titles and descriptions
        title_embs = self.text_model.encode(list(titles), show_progress_bar=False)
        desc_embs = self.text_model.encode(list(descs), show_progress_bar=False)
        
        # Save to database
        rows = []
        for i in range(len(batch)):
            rows.append((
                urls[i], 
                titles[i], 
                descs[i], 
                json.dumps(images_lists[i]),
                title_embs[i].astype(np.float32).tobytes(),
                desc_embs[i].astype(np.float32).tobytes()
            ))
        
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.executemany("""
            INSERT OR REPLACE INTO pages
            (url, title, description, image_links, title_emb, desc_emb)
            VALUES (?, ?, ?, ?, ?, ?)
        """, rows)
        conn.commit()
        conn.close()
        
        print(f"✓ Saved {len(batch)} pages")
        return len(batch)
    
    def load_data(self):
        """Load all data from database into memory"""
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT url, title, description, image_links, title_emb, desc_emb FROM pages")
        rows = c.fetchall()
        conn.close()
        
        if not rows:
            print("⚠️  No data indexed yet")
            self.pages_data = []
            return
        
        self.pages_data = []
        for url, title, desc, img_links, title_emb_blob, desc_emb_blob in rows:
            title_emb = np.frombuffer(title_emb_blob, dtype=np.float32)
            desc_emb = np.frombuffer(desc_emb_blob, dtype=np.float32)
            
            self.pages_data.append({
                'url': url,
                'title': title,
                'description': desc,
                'image_links': json.loads(img_links) if img_links else [],
                'title_emb': title_emb,
                'desc_emb': desc_emb
            })
        
        print(f"✓ Loaded {len(self.pages_data)} pages")
    
    def search(self, query_text, top_k=20, threshold=0.25):
        """Simple cosine similarity search (text only)"""
        if not query_text:
            raise ValueError("Query text is required")
        
        if not self.pages_data:
            return []
        
        # Encode query
        query_emb = self.text_model.encode([query_text], show_progress_bar=False)[0]
        
        # Calculate similarities
        results = []
        for page in self.pages_data:
            # Similarity with title
            title_sim = cosine_similarity(
                query_emb.reshape(1, -1), 
                page['title_emb'].reshape(1, -1)
            )[0][0]
            
            # Similarity with description
            desc_sim = cosine_similarity(
                query_emb.reshape(1, -1), 
                page['desc_emb'].reshape(1, -1)
            )[0][0]
            
            # Take max similarity
            similarity = max(title_sim, desc_sim)
            
            if similarity > threshold:
                results.append({
                    'url': page['url'],
                    'title': page['title'],
                    'description': page['description'],
                    'image_links': page['image_links'],
                    'similarity': float(similarity)
                })
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:top_k]

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
    indexer.load_data()  # Reload data after adding
    return count