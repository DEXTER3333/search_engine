#!/usr/bin/env python3
"""
Fast Search Engine - Main Entry Point
Optimized for speed with simple cosine similarity
"""

import argparse
import sys
import os

def crawl_sites(urls=None):
    """Run the crawler to index websites"""
    from indexer_optimized import get_indexer
    from crawler_optimized import run_crawler
    
    # Si aucune URL fournie, lire depuis urls.txt
    if urls is None:
        if os.path.exists('urls.txt'):
            print("📖 Reading URLs from urls.txt...")
            with open('urls.txt', 'r') as f:
                urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        else:
            # URLs par défaut
            urls = [
                "https://www.wikipedia.org",
                "https://www.github.com",
                "https://www.stackoverflow.com",
                "https://www.reddit.com",
                "https://www.medium.com"
            ]
    
    print(f"Starting crawler with {len(urls)} seed URLs...")
    print("This will crawl up to 50 pages per site")
    print("-" * 60)
    
    # Initialize indexer
    get_indexer()
    
    # Run crawler
    run_crawler(urls)
    
    print("-" * 60)
    print("Crawling complete! Reloading data...")
    
    # Reload data
    indexer = get_indexer()
    indexer.load_data()
    
    print(f"✓ Total pages indexed: {len(indexer.pages_data)}")
    print("✓ Ready to serve queries!")

def start_server(host="0.0.0.0", port=8000):
    """Start the FastAPI server"""
    import uvicorn
    from api_server import app
    
    print(f"Starting Fast Search Engine API on http://{host}:{port}")
    print("Press Ctrl+C to stop")
    print("-" * 60)
    
    uvicorn.run(app, host=host, port=port, log_level="info")

def test_search(query=None):
    """Test search functionality"""
    from indexer_optimized import get_indexer
    
    indexer = get_indexer()
    
    if not indexer.pages_data:
        print("No data indexed yet. Run crawl first.")
        return
    
    if query is None:
        query = input("Enter search query: ")
    
    print(f"\nSearching for: '{query}'")
    print("-" * 60)
    
    import time
    start = time.time()
    results = indexer.search(query_text=query, top_k=10, threshold=0.3)
    elapsed = time.time() - start
    
    print(f"Found {len(results)} results in {elapsed:.3f}s\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']}")
        print(f"   URL: {result['url']}")
        print(f"   Similarity: {result['similarity']:.2%}")
        print(f"   {result['description'][:100]}...")
        print()

def show_stats():
    """Show database statistics"""
    from indexer_optimized import get_indexer
    
    indexer = get_indexer()
    
    print("Search Engine Statistics")
    print("=" * 60)
    print(f"Total pages indexed: {len(indexer.pages_data)}")
    
    if indexer.pages_data:
        print(f"Index type: In-memory cosine similarity")
        print(f"Embeddings: Text only")
        print(f"Text model: all-MiniLM-L6-v2")
    else:
        print("No data indexed yet.")
    
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description="Fast Search Engine - Semantic search with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 main_optimized.py crawl                    # Crawl from urls.txt
  python3 main_optimized.py crawl --urls custom.txt  # Crawl from custom file
  python3 main_optimized.py server                   # Start API server
  python3 main_optimized.py search "machine learning" # Test search
  python3 main_optimized.py stats                    # Show statistics
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Crawl command
    crawl_parser = subparsers.add_parser('crawl', help='Crawl and index websites')
    crawl_parser.add_argument('--urls', type=str, help='File with URLs (one per line)')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start API server')
    server_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind')
    server_parser.add_argument('--port', type=int, default=8000, help='Port to bind')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Test search query')
    search_parser.add_argument('query', type=str, nargs='?', help='Search query')
    
    # Stats command
    subparsers.add_parser('stats', help='Show database statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'crawl':
            urls = None
            if args.urls:
                with open(args.urls, 'r') as f:
                    urls = [line.strip() for line in f if line.strip()]
            crawl_sites(urls)
        
        elif args.command == 'server':
            start_server(args.host, args.port)
        
        elif args.command == 'search':
            test_search(args.query)
        
        elif args.command == 'stats':
            show_stats()
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()