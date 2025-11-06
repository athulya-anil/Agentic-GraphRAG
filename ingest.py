#!/usr/bin/env python3
"""
Agentic GraphRAG - Document Ingestion CLI

This script allows you to ingest your own documents into the knowledge graph.
Supports various file formats: .txt, .md, .pdf, .docx

Usage:
    # Ingest a single file
    python ingest.py --file path/to/document.txt

    # Ingest multiple files
    python ingest.py --file doc1.txt --file doc2.pdf --file doc3.md

    # Ingest all files in a directory
    python ingest.py --dir path/to/documents/

    # Ingest with custom options
    python ingest.py --dir data/raw/ --no-schema-inference --no-metadata

Author: Agentic GraphRAG Team
"""

import sys
import argparse
from pathlib import Path
from typing import List
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import get_ingestion_pipeline


def load_document(file_path: Path) -> str:
    """Load document content from various file formats."""
    suffix = file_path.suffix.lower()

    try:
        if suffix in ['.txt', '.md']:
            # Plain text files
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

        elif suffix == '.pdf':
            # PDF files - requires PyPDF2 or pypdf
            try:
                import pypdf
                with open(file_path, 'rb') as f:
                    reader = pypdf.PdfReader(f)
                    text = []
                    for page in reader.pages:
                        text.append(page.extract_text())
                    return '\n\n'.join(text)
            except ImportError:
                print(f"⚠️  Warning: pypdf not installed. Install with: pip install pypdf")
                print(f"   Skipping {file_path.name}")
                return None

        elif suffix in ['.docx', '.doc']:
            # Word documents - requires python-docx
            try:
                from docx import Document
                doc = Document(file_path)
                text = []
                for para in doc.paragraphs:
                    text.append(para.text)
                return '\n\n'.join(text)
            except ImportError:
                print(f"⚠️  Warning: python-docx not installed. Install with: pip install python-docx")
                print(f"   Skipping {file_path.name}")
                return None

        else:
            print(f"⚠️  Warning: Unsupported file format: {suffix}")
            print(f"   Supported formats: .txt, .md, .pdf, .docx")
            print(f"   Skipping {file_path.name}")
            return None

    except Exception as e:
        print(f"❌ Error reading {file_path.name}: {e}")
        return None


def collect_files_from_directory(directory: Path, recursive: bool = False) -> List[Path]:
    """Collect all supported document files from a directory."""
    supported_extensions = {'.txt', '.md', '.pdf', '.docx', '.doc'}
    files = []

    if recursive:
        for ext in supported_extensions:
            files.extend(directory.rglob(f'*{ext}'))
    else:
        for ext in supported_extensions:
            files.extend(directory.glob(f'*{ext}'))

    return sorted(files)


def main():
    """Run document ingestion from command line."""
    parser = argparse.ArgumentParser(
        description='Ingest documents into Agentic GraphRAG knowledge graph',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a single file
  python ingest.py --file document.txt

  # Ingest multiple files
  python ingest.py --file doc1.txt --file doc2.pdf

  # Ingest all files in a directory
  python ingest.py --dir data/raw/

  # Ingest recursively with custom schema
  python ingest.py --dir data/raw/ --recursive --schema-path custom_schema.json
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--file', '-f',
        type=str,
        action='append',
        dest='files',
        help='Path to a document file (can be specified multiple times)'
    )
    input_group.add_argument(
        '--dir', '-d',
        type=str,
        help='Path to directory containing documents'
    )

    # Directory options
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Recursively search directory for documents'
    )

    # Processing options
    parser.add_argument(
        '--schema-path',
        type=str,
        default='data/processed/schema.json',
        help='Path to schema file (default: data/processed/schema.json)'
    )
    parser.add_argument(
        '--no-schema-inference',
        action='store_true',
        help='Skip automatic schema inference (use existing schema)'
    )
    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='Skip metadata enrichment (faster but less informative)'
    )
    parser.add_argument(
        '--no-auto-refine',
        action='store_true',
        help='Disable automatic schema refinement'
    )

    # Output options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except errors'
    )

    args = parser.parse_args()

    # Collect files to process
    file_paths = []
    if args.files:
        file_paths = [Path(f) for f in args.files]
    elif args.dir:
        dir_path = Path(args.dir)
        if not dir_path.exists():
            print(f"❌ Error: Directory not found: {args.dir}")
            sys.exit(1)
        if not dir_path.is_dir():
            print(f"❌ Error: Not a directory: {args.dir}")
            sys.exit(1)

        file_paths = collect_files_from_directory(dir_path, args.recursive)
        if not file_paths:
            print(f"⚠️  No supported documents found in: {args.dir}")
            print(f"   Supported formats: .txt, .md, .pdf, .docx")
            sys.exit(0)

    # Verify files exist
    valid_files = []
    for fp in file_paths:
        if not fp.exists():
            print(f"⚠️  Warning: File not found: {fp}")
        elif not fp.is_file():
            print(f"⚠️  Warning: Not a file: {fp}")
        else:
            valid_files.append(fp)

    if not valid_files:
        print("❌ Error: No valid files to process")
        sys.exit(1)

    # Print header
    if not args.quiet:
        print("=" * 70)
        print("  AGENTIC GRAPHRAG - DOCUMENT INGESTION")
        print("=" * 70)
        print(f"\n📚 Found {len(valid_files)} document(s) to ingest:")
        for fp in valid_files:
            print(f"   • {fp.name} ({fp.stat().st_size / 1024:.1f} KB)")

    # Load documents
    if not args.quiet:
        print(f"\n📖 Loading documents...")

    documents = []
    doc_metadata = []

    for fp in valid_files:
        if args.verbose:
            print(f"   Loading {fp.name}...")

        content = load_document(fp)
        if content and content.strip():
            documents.append(content)
            doc_metadata.append({
                'filename': fp.name,
                'path': str(fp),
                'size_bytes': fp.stat().st_size
            })
        else:
            print(f"⚠️  Warning: Empty or unreadable file: {fp.name}")

    if not documents:
        print("❌ Error: No documents successfully loaded")
        sys.exit(1)

    if not args.quiet:
        print(f"✅ Loaded {len(documents)} document(s)")
        total_chars = sum(len(doc) for doc in documents)
        print(f"   Total content: {total_chars:,} characters")

    # Initialize ingestion pipeline
    if not args.quiet:
        print(f"\n🔧 Initializing ingestion pipeline...")
        print(f"   • Schema path: {args.schema_path}")
        print(f"   • Schema inference: {'disabled' if args.no_schema_inference else 'enabled'}")
        print(f"   • Metadata enrichment: {'disabled' if args.no_metadata else 'enabled'}")
        print(f"   • Auto-refine schema: {'disabled' if args.no_auto_refine else 'enabled'}")

    try:
        pipeline = get_ingestion_pipeline(
            schema_path=Path(args.schema_path),
            auto_refine_schema=not args.no_auto_refine
        )
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Ingest documents
    if not args.quiet:
        print(f"\n🔄 Running ingestion pipeline...")
        if not args.no_schema_inference:
            print(f"   • Inferring graph schema")
        print(f"   • Extracting entities")
        print(f"   • Identifying relationships")
        print(f"   • Building knowledge graph")
        print(f"   • Creating vector embeddings")

    start_time = time.time()

    try:
        results = pipeline.ingest_documents(
            documents,
            infer_schema=not args.no_schema_inference,
            enrich_metadata=not args.no_metadata
        )
    except Exception as e:
        print(f"\n❌ Error during ingestion: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    duration = time.time() - start_time

    # Print results
    if not args.quiet:
        print(f"\n✅ Ingestion complete in {duration:.2f}s")
        print(f"\n📊 Results:")
        print(f"   • Documents processed: {results.get('documents_processed', 0)}")
        print(f"   • Entities extracted: {results.get('entities_extracted', 0)}")
        print(f"   • Relations extracted: {results.get('relations_extracted', 0)}")
        print(f"   • Neo4j nodes created: {results.get('nodes_created', 0)}")
        print(f"   • Neo4j edges created: {results.get('edges_created', 0)}")

        # Show schema summary
        if not args.no_schema_inference:
            try:
                from src.agents import get_schema_agent
                schema_agent = get_schema_agent()
                print(f"\n📋 Schema Summary:")
                summary = schema_agent.get_schema_summary()
                print(summary)
            except Exception as e:
                if args.verbose:
                    print(f"⚠️  Could not load schema summary: {e}")

        print(f"\n" + "=" * 70)
        print("✅ Ingestion successful!")
        print(f"\n💡 Next steps:")
        print(f"   • Query your knowledge graph with: python query.py")
        print(f"   • View in Neo4j Browser: http://localhost:7474")
        print(f"   • Run full demo: python demo.py")
        print("=" * 70)

    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Ingestion interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
