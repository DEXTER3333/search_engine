from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import sys
import os

# Import your optimized indexer
sys.path.append(os.path.dirname(__file__))
from indexer_optimized import get_indexer

app = FastAPI(title="Fast Search Engine API")

# Enable CORS for frontend
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
    threshold: Optional[float] = 0.25

class SearchResponse(BaseModel):
    results: List[dict]
    count: int
    query_time: float

@app.on_event("startup")
async def startup_event():
    """Initialize indexer on startup"""
    print("=" * 60)
    print("Initializing search engine...")
    try:
        indexer = get_indexer()
        print(f"✓ Search engine ready! {len(indexer.pages_data)} pages loaded")
    except Exception as e:
        print(f"✗ Error initializing: {e}")
    print("=" * 60)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML"""
    try:
        with open("frontend.html", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return """
        <html>
            <body style="font-family: Arial; text-align: center; padding: 50px;">
                <h1>🔍 Search Engine API</h1>
                <p>API is running!</p>
                <p><a href="/docs">View API Documentation</a></p>
            </body>
        </html>
        """

@app.post("/search", response_model=SearchResponse)
async def search_text(request: SearchRequest):
    """Search by text query"""
    import time
    start = time.time()
    
    try:
        indexer = get_indexer()
        
        if not indexer.pages_data:
            raise HTTPException(
                status_code=503,
                detail="No data indexed yet. Database is empty."
            )
        
        results = indexer.search(
            query_text=request.query,
            top_k=request.top_k,
            threshold=request.threshold
        )
        
        query_time = time.time() - start
        
        return SearchResponse(
            results=results,
            count=len(results),
            query_time=round(query_time, 3)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    try:
        indexer = get_indexer()
        return {
            "total_pages": len(indexer.pages_data),
            "status": "ready" if indexer.pages_data else "empty",
            "message": "API is running"
        }
    except Exception as e:
        return {
            "total_pages": 0,
            "status": "error",
            "error": str(e)
        }

@app.post("/rebuild-index")
async def rebuild_index():
    """Rebuild indexes from database"""
    try:
        indexer = get_indexer()
        indexer.load_data()
        return {
            "message": "Data reloaded successfully", 
            "total_pages": len(indexer.pages_data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    
    print("\n" + "=" * 60)
    print("Starting Fast Search Engine API")
    print("=" * 60)
    print(f"API available at: http://0.0.0.0:{port}")
    print(f"Documentation at: http://0.0.0.0:{port}/docs")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")