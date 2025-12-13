from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import sys  
import os

sys.path.append(os.path.dirname(__file__))
from indexer_optimized import get_indexer

app = FastAPI(title="🧠 ML/DL Search Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    threshold: Optional[float] = 0.1  # Seuil encore plus bas pour Fast/Balanced
    mode: Optional[str] = "balanced"
    use_expansion: Optional[bool] = True

class SearchResponse(BaseModel):
    results: List[dict]
    count: int
    query_time: float
    mode: str

@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print("🚀 Initialisation du moteur ML/DL...")
    try:
        indexer = get_indexer()
        print(f"✓ Moteur prêt: {len(indexer.pages_data)} pages")
    except Exception as e:
        print(f"✗ Erreur: {e}")
    print("=" * 60)

@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open("frontend.html", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return """
        <html>
            <body style="font-family: Arial; text-align: center; padding: 50px;">
                <h1>🧠 ML/DL Search Engine API</h1>
                <p>API is running!</p>
            </body>
        </html>
        """

@app.post("/search", response_model=SearchResponse)
async def search_text(request: SearchRequest):
    import time
    start = time.time()
    
    try:
        indexer = get_indexer()
        
        if not indexer.pages_data:
            raise HTTPException(
                status_code=503,
                detail="Base de données vide"
            )
        
        # Configuration selon le mode
        mode_config = {
            "fast": {"rerank": False, "expand": False},
            "balanced": {"rerank": False, "expand": True},
            "precise": {"rerank": True, "expand": True}
        }
        
        mode = request.mode.lower()
        if mode not in mode_config:
            mode = "balanced"
        
        config = mode_config[mode]
        
        results = indexer.search(
            query_text=request.query,
            top_k=request.top_k,
            threshold=request.threshold,
            use_reranking=config["rerank"],
            use_expansion=config["expand"] and request.use_expansion
        )
        
        query_time = time.time() - start
        
        return SearchResponse(
            results=results,
            count=len(results),
            query_time=round(query_time, 3),
            mode=mode
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    print("\n" + "=" * 60)
    print("🧠 Starting ML/DL Search Engine API")
    print("=" * 60)
    print(f"URL: http://0.0.0.0:{port}")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")