"""
Talk Endee - Main CLI Interface
Semantic Search + RAG System powered by Endee Vector Database
"""

import argparse
import sys
from pathlib import Path
from src.config import Config
from src.ingest import IngestionPipeline
from src.retriever import RAGRetriever
from src.generator import AnswerGenerator

def cmd_ingest(args):
    """Handle ingest command"""
    try:
        pipeline = IngestionPipeline()
        
        if args.file:
            result = pipeline.ingest_file(args.file)
            print(f"\n✓ Successfully ingested: {result['file']}")
            print(f"  - Chunks created: {result['chunks']}")
            print(f"  - Vectors stored: {result['vectors_stored']}")
        
        elif args.directory:
            results = pipeline.ingest_directory(args.directory, args.extension)
            print(f"\n✓ Ingestion complete!")
            total_chunks = sum(r.get('chunks', 0) for r in results)
            total_vectors = sum(r.get('vectors_stored', 0) for r in results)
            print(f"  - Files processed: {len(results)}")
            print(f"  - Total chunks: {total_chunks}")
            print(f"  - Total vectors stored: {total_vectors}")
    
    except Exception as e:
        print(f"✗ Error during ingestion: {e}", file=sys.stderr)
        sys.exit(1)

def cmd_query(args):
    """Handle query command"""
    try:
        retriever = RAGRetriever()
        retrieved_docs = retriever.retrieve(args.query, top_k=args.top_k)
        
        if not retrieved_docs:
            print("No relevant documents found.")
            return
        
        print(f"\n{'='*60}")
        print(f"Query: {args.query}")
        print(f"{'='*60}\n")
        
        print(f"Retrieved {len(retrieved_docs)} relevant documents:\n")
        for doc in retrieved_docs:
            print(f"[{doc['rank']}] {doc['source']} (Score: {doc['score']:.3f})")
            print(f"    {doc['text'][:100]}...")
            print()
        
        if not args.search_only:
            print(f"{'='*60}")
            print("Generating answer...")
            print(f"{'='*60}\n")
            
            context = retriever.format_context(retrieved_docs)
            generator = AnswerGenerator()
            result = generator.generate(args.query, context)
            
            print(f"Answer:\n{result['answer']}\n")
            print(f"{'='*60}")
            print(f"Tokens used: {result['usage']['total_tokens']}")
            print(f"{'='*60}")
    
    except Exception as e:
        print(f"✗ Error during query: {e}", file=sys.stderr)
        sys.exit(1)

def cmd_info(args):
    """Handle info command"""
    try:
        from src.endee_client import EndeeClient
        client = EndeeClient()
        
        # list_indices() returns dict with 'indexes' key containing list of index dicts
        response = client.list_indices()
        indices_list = response.get("indexes", [])
        
        if not indices_list:
            print("\nNo indices found.\n")
            return
        
        print(f"\nEndee Indices:\n")
        for index_dict in indices_list:
            idx_name = index_dict.get("name", "unknown")
            total_elements = index_dict.get("total_elements", 0)
            dimension = index_dict.get("dimension", 0)
            space_type = index_dict.get("space_type", "unknown")
            precision = index_dict.get("precision", "unknown")
            
            print(f"Index: {idx_name}")
            print(f"  - Dimension: {dimension}")
            print(f"  - Total elements: {total_elements}")
            print(f"  - Space type: {space_type}")
            print(f"  - Precision: {precision}")
            print()
    
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Talk Endee - Semantic Search + RAG with Endee Vector Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest documents
  python main.py ingest --file data/sample_docs/document.txt
  python main.py ingest --directory data/sample_docs

  # Query the system
  python main.py query "What is semantic search?"
  python main.py query "How does RAG work?" --top-k 3

  # Show system info
  python main.py info
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into Endee")
    ingest_group = ingest_parser.add_mutually_exclusive_group(required=True)
    ingest_group.add_argument("--file", help="Path to a single document file")
    ingest_group.add_argument("--directory", help="Path to directory with documents")
    ingest_parser.add_argument("--extension", default=".txt", help="File extension to look for (default: .txt)")
    ingest_parser.set_defaults(func=cmd_ingest)
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("query", help="Query text")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve (default: 5)")
    query_parser.add_argument("--search-only", action="store_true", help="Only search, don't generate answer")
    query_parser.set_defaults(func=cmd_query)
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show system information")
    info_parser.set_defaults(func=cmd_info)
    
    args = parser.parse_args()
    
    # Validate config
    try:
        Config.validate()
    except ValueError as e:
        print(f"✗ Configuration Error: {e}", file=sys.stderr)
        print(f"\nPlease create a .env file. Use .env.example as a template.")
        sys.exit(1)
    
    # Execute command
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    args.func(args)

if __name__ == "__main__":
    main()
