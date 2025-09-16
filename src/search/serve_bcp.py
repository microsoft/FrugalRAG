#!/usr/bin/env python3
"""
Simple, efficient Flask server for BrowseComp-Plus searchers.
Alternative to the slow MCP server with 307 errors.
"""

import os
import sys
import argparse
from functools import lru_cache
from typing import Dict, List, Any, Optional

from flask import Flask, request, jsonify, Response
from dotenv import load_dotenv
import json

# Add the bcplus searcher path
current_dir = os.path.dirname(os.path.abspath(__file__))
bcplus_path = os.path.join(current_dir, "bcplus", "searcher")
if bcplus_path not in sys.path:
    sys.path.insert(0, bcplus_path)

from searchers import SearcherType

# Load environment variables
load_dotenv()

# Global variables
app = Flask(__name__)
searcher = None
tokenizer = None
counter = {"api": 0}
snippet_max_tokens = 512

def create_searcher(searcher_type: str, args) -> Any:
    """Create and initialize a searcher based on type."""
    searcher_class = SearcherType.get_searcher_class(searcher_type)
    return searcher_class(args)

def create_tokenizer(snippet_max_tokens: int) -> Optional[Any]:
    """Create tokenizer for snippet truncation if needed."""
    if snippet_max_tokens and snippet_max_tokens > 0:
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        except Exception as e:
            print(f"[WARNING] Failed to load tokenizer: {e}")
            return None
    return None

def format_search_results_mcp_style(results: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """Format search results in MCP-compatible format with snippet handling."""
    global tokenizer, snippet_max_tokens
    
    # Apply snippet truncation if tokenizer is available
    if snippet_max_tokens and snippet_max_tokens > 0 and tokenizer:
        for result in results:
            text = result.get("text", "")
            try:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > snippet_max_tokens:
                    truncated_tokens = tokens[:snippet_max_tokens]
                    result["snippet"] = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                else:
                    result["snippet"] = text
            except Exception as e:
                print(f"[WARNING] Tokenization failed: {e}")
                result["snippet"] = text
    else:
        for result in results:
            result["snippet"] = result.get("text", "")
    
    # Format results in MCP style
    formatted_results = []
    for result in results:
        formatted_result = {
            "docid": result["docid"],
            "snippet": result["snippet"]
        }
        
        # Include score if available
        if "score" in result and result["score"] is not None:
            formatted_result["score"] = result["score"]
        
        formatted_results.append(formatted_result)
    
    return formatted_results

@lru_cache(maxsize=10000)
def cached_search(query: str, k: int) -> List[Dict[str, Any]]:
    """Cached search function returning MCP-compatible format."""
    if not searcher:
        raise RuntimeError("Searcher not initialized")
    
    print(f"[SEARCH] Query: {query[:100]}{'...' if len(query) > 100 else ''}")
    
    # Limit k to reasonable bounds
    k = min(max(int(k), 1), 100)
    
    try:
        results = searcher.search(query, k=k)
        
        # Format results in MCP-compatible style
        formatted_results = format_search_results_mcp_style(results, k)
        
        return formatted_results
        
    except Exception as e:
        print(f"[ERROR] Search failed: {e}")
        return []

@app.route("/api/search", methods=["GET", "POST"])
def api_search():
    """Main search endpoint returning MCP-compatible format."""
    counter["api"] += 1
    
    if request.method == "GET":
        query = request.args.get("query", "").strip()
        k = request.args.get("k", 10)
    elif request.method == "POST":
        data = request.get_json() or {}
        query = data.get("query", "").strip()
        k = data.get("k", 10)
    else:
        return jsonify({"error": "Method not allowed"}), 405
    
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    try:
        k = int(k)
    except (ValueError, TypeError):
        k = 10
    
    print(f"[API] Request #{counter['api']}: k={k}")
    
    # Return results directly in MCP-compatible format
    results = cached_search(query, k)
    return jsonify(results)

@app.route("/api/get_document", methods=["GET", "POST"])
def api_get_document():
    """Get full document by docid."""
    if request.method == "GET":
        docid = request.args.get("docid", "").strip()
    elif request.method == "POST":
        data = request.get_json() or {}
        docid = data.get("docid", "").strip()
    else:
        return jsonify({"error": "Method not allowed"}), 405
    
    if not docid:
        return jsonify({"error": "docid parameter is required"}), 400
    
    if not searcher:
        return jsonify({"error": "Searcher not initialized"}), 500
    
    try:
        doc = searcher.get_document(docid)
        if doc is None:
            return jsonify({"error": f"Document with docid '{docid}' not found"}), 404
        
        return jsonify({
            "docid": docid,
            "found": True,
            "document": doc
        })
        
    except Exception as e:
        print(f"[ERROR] Get document failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "searcher_type": searcher.search_type if searcher else None,
        "searcher_initialized": searcher is not None,
        "total_requests": counter["api"]
    })

@app.route("/", methods=["GET"])
def root():
    """Root endpoint with basic info."""
    return jsonify({
        "service": "BrowseComp-Plus Search Server",
        "searcher_type": searcher.search_type if searcher else None,
        "endpoints": {
            "search": "/api/search",
            "get_document": "/api/get_document", 
            "health": "/api/health"
        },
        "usage": {
            "search": "GET/POST /api/search?query=YOUR_QUERY&k=10",
            "get_document": "GET/POST /api/get_document?docid=YOUR_DOCID"
        }
    })

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="BrowseComp-Plus Search Server")
    
    parser.add_argument(
        "--searcher-type",
        choices=SearcherType.get_choices(),
        required=True,
        help=f"Type of searcher to use: {', '.join(SearcherType.get_choices())}"
    )
    
    parser.add_argument(
        "--snippet-max-tokens",
        type=int,
        default=512,
        help="Number of tokens to include for each document snippet in search results using Qwen/Qwen3-0.6B tokenizer (default: 512). Set to -1 to disable."
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Port to run the server on (default: 8000, or PORT env var)"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )
    
    # Parse searcher type first to get the right searcher class
    temp_args, _ = parser.parse_known_args()
    searcher_class = SearcherType.get_searcher_class(temp_args.searcher_type)
    
    # Add searcher-specific arguments
    searcher_class.parse_args(parser)
    
    return parser.parse_args()

def main():
    """Main entry point."""
    global searcher, tokenizer, snippet_max_tokens
    
    try:
        args = parse_arguments()
        
        # Set snippet max tokens
        snippet_max_tokens = args.snippet_max_tokens if args.snippet_max_tokens >= 0 else None
        
        # Initialize tokenizer if snippet truncation is enabled
        if snippet_max_tokens and snippet_max_tokens > 0:
            print(f"[INIT] Initializing tokenizer for snippet truncation (max_tokens={snippet_max_tokens})...")
            tokenizer = create_tokenizer(snippet_max_tokens)
            if tokenizer:
                print("[INIT] Tokenizer initialized successfully")
            else:
                print("[INIT] Tokenizer failed to initialize, snippets will not be truncated")
        
        # Set HF environment variables if provided
        if hasattr(args, 'hf_token') and args.hf_token:
            os.environ['HF_TOKEN'] = args.hf_token
            os.environ['HUGGINGFACE_HUB_TOKEN'] = args.hf_token
        
        if hasattr(args, 'hf_home') and args.hf_home:
            os.environ['HF_HOME'] = args.hf_home
        
        print(f"[INIT] Initializing {args.searcher_type} searcher...")
        searcher = create_searcher(args.searcher_type, args)
        print(f"[INIT] Searcher initialized successfully: {searcher.search_type}")
        
        # Warm up the searcher with a test query
        print("[INIT] Warming up searcher...")
        try:
            warmup_results = searcher.search("test query", k=1)
            print(f"[INIT] Warmup successful, got {len(warmup_results)} results")
        except Exception as e:
            print(f"[INIT] Warmup failed (non-critical): {e}")
        
        print(f"[SERVER] Starting Flask server on {args.host}:{args.port}")
        print(f"[SERVER] Output format: MCP-compatible (docid, score, snippet)")
        print(f"[SERVER] Snippet truncation: {'Enabled' if snippet_max_tokens else 'Disabled'}")
        print(f"[SERVER] Endpoints available:")
        print(f"  - GET/POST {args.host}:{args.port}/api/search")
        print(f"  - GET/POST {args.host}:{args.port}/api/get_document")
        print(f"  - GET     {args.host}:{args.port}/api/health")
        print(f"[SERVER] Cache size limit: 10,000 queries")
        
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
        
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down gracefully...")
    except Exception as e:
        print(f"[ERROR] Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
