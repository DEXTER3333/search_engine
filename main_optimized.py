#!/usr/bin/env python3
"""
main_optimized.py - Point d'entrée simplifié
"""

import argparse
import sys
import os

def crawl_sites(urls=None):
    """Run the crawler to index websites"""
    from indexer_optimized import get_indexer
    from crawler_optimized import run_crawler
    
    if urls is None:
        if os.path.exists('urls.txt'):
            print("📖 Reading URLs from urls.txt...")
            with open('urls.txt', 'r') as f:
                urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        else:
            urls = [
                "https://www.wikipedia.org",
                "https://www.github.com",
                "https://www.stackoverflow.com",
                "https://www.reddit.com",
                "https://www.medium.com"
            ]
    
    print(f"🚀 Starting ML crawler with {len(urls)} seed URLs...")
    print("   Crawling up to 50 pages per site")
    print("-" * 70)
    
    get_indexer()
    run_crawler(urls)
    
    print("-" * 70)
    print("✅ Crawling complete! Reloading data...")
    
    indexer = get_indexer()
    indexer.load_data()
    
    print(f"✓ Total pages indexed: {len(indexer.pages_data)}")
    print("✓ Ready to serve queries with ML/DL!")

def start_server(host="0.0.0.0", port=8000):
    """Start the FastAPI ML server"""
    import uvicorn
    from api_server import app
    
    print("=" * 70)
    print("🧠 Starting ML/DL Search Engine API")
    print("=" * 70)
    print(f"Server: http://{host}:{port}")
    print(f"Frontend: http://{host}:{port}/")
    print("-" * 70)
    print("Modes disponibles:")
    print("  • Fast (~50ms): Bi-encoder uniquement")
    print("  • Balanced (~80ms): Bi-encoder + BM25 fusion")
    print("  • Precise (~150ms): + Cross-encoder reranking")
    print("=" * 70)
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(app, host=host, port=port, log_level="info")

def test_search(query=None, mode="balanced"):
    """Test search"""
    from indexer_optimized import get_indexer
    
    indexer = get_indexer()
    
    if not indexer.pages_data:
        print("❌ No data indexed yet. Run 'crawl' first.")
        return
    
    if query is None:
        query = input("Enter search query: ")
    
    print("\n" + "=" * 70)
    print(f"🔍 Searching: '{query}'")
    print(f"Mode: {mode.upper()}")
    print("=" * 70)
    
    import time
    
    use_reranking = mode == "precise"
    use_expansion = mode in ["balanced", "precise"]
    
    start = time.time()
    results = indexer.search(
        query_text=query,
        top_k=10,
        threshold=0.1,  # Seuil plus bas
        use_reranking=use_reranking,
        use_expansion=use_expansion
    )
    elapsed = time.time() - start
    
    print(f"\n✓ Found {len(results)} results in {elapsed*1000:.1f}ms")
    print("-" * 70 + "\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title'][:70]}")
        print(f"   📊 Score: {result['similarity']:.3f} | PageRank: {result['page_rank']:.3f}")
        print(f"   🔗 {result['url'][:80]}")
        print(f"   📝 {result['description'][:150]}...")
        print()

def main():
    parser = argparse.ArgumentParser(
        description="ML/DL Search Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 main_optimized.py crawl
  python3 main_optimized.py server
  python3 main_optimized.py search "machine learning" --mode balanced
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    crawl_parser = subparsers.add_parser('crawl', help='Crawl and index websites')
    crawl_parser.add_argument('--urls', type=str, help='File with URLs (one per line)')
    
    server_parser = subparsers.add_parser('server', help='Start ML API server')
    server_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind')
    server_parser.add_argument('--port', type=int, default=8000, help='Port to bind')
    
    search_parser = subparsers.add_parser('search', help='Test search with ML')
    search_parser.add_argument('query', type=str, nargs='?', help='Search query')
    search_parser.add_argument('--mode', type=str, default='balanced',
                             choices=['fast', 'balanced', 'precise'],
                             help='Search mode (default: balanced)')
    
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
            test_search(args.query, args.mode)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()