#!/usr/bin/env python3
"""
Test script for text-based similarity highlighting functionality.
This script demonstrates how to use the new CLIP-based text query feature.
"""

import os
import sys
import subprocess
import argparse

def test_text_similarity():
    """Test the text-based similarity highlighting functionality."""
    
    # Example queries to test
    test_queries = [
        "chair",
        "table",
        "tree",
        "building",
        "car",
        "person",
        "sky",
        "grass",
        "window",
        "door",
        "book",
        "computer",
        "lamp",
        "flower"
    ]
    
    print("🧪 Text-based Similarity Highlighting Test")
    print("=" * 50)
    print("\nThis script will help you test the new text-based similarity highlighting feature.")
    print("You can search for objects or concepts in your featurized point cloud using natural language.")
    print("\nExample queries:", ", ".join(test_queries[:8]))
    print()
    
    return test_queries

def run_visualization_with_text_query(pointcloud_dir, text_query, **kwargs):
    """Run the visualization script with a text query."""
    
    # Build the command
    cmd = [
        "python", "rerun/visualize_featurized_pointcloud.py", # Changed
        pointcloud_dir,
        "--text-query", text_query,
        "--enable-highlights",
        "--feature-type", "both",
        "--point-size", "0.015"
    ]
    
    # Add optional arguments
    if kwargs.get('mode'):
        cmd.extend(["--mode", kwargs['mode']])
    if kwargs.get('remote_host'):
        cmd.extend(["--remote-host", kwargs['remote_host']])
    if kwargs.get('remote_port'):
        cmd.extend(["--remote-port", str(kwargs['remote_port'])])
    if kwargs.get('top_k'):
        cmd.extend(["--text-similarity-top-k", str(kwargs['top_k'])])
    if kwargs.get('threshold'):
        cmd.extend(["--text-similarity-threshold", str(kwargs['threshold'])])
    if kwargs.get('clip_model'):
        cmd.extend(["--clip-model-version", kwargs['clip_model']])
    if kwargs.get('create_mesh'):
        cmd.append("--create-mesh")
    
    print(f"🚀 Running: {' '.join(cmd)}")
    print()
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running visualization: {e}")
    except KeyboardInterrupt:
        print("\n⏹️  Visualization interrupted by user")

def main():
    parser = argparse.ArgumentParser(description="Test text-based similarity highlighting")
    parser.add_argument("pointcloud_dir", 
                       help="Directory containing featurized pointcloud .pt files")
    parser.add_argument("--query", 
                       help="Text query to search for (if not provided, will prompt)")
    parser.add_argument("--mode", choices=["serve", "web", "save"], default="serve",
                       help="Visualization mode")
    parser.add_argument("--remote-host", help="Remote host for streaming")
    parser.add_argument("--remote-port", type=int, default=9878, help="Remote port")
    parser.add_argument("--top-k", type=int, default=100, 
                       help="Number of top similar points to highlight")
    parser.add_argument("--threshold", type=float, default=0.2,
                       help="Similarity threshold (lower = more results)")
    parser.add_argument("--clip-model", choices=["ViT-B/32", "ViT-L/14"], default="ViT-B/32",
                       help="CLIP model version")
    parser.add_argument("--create-mesh", action="store_true",
                       help="Create surface mesh")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive mode - prompt for multiple queries")
    
    args = parser.parse_args()
    
    # Check if pointcloud directory exists
    if not os.path.exists(args.pointcloud_dir):
        print(f"❌ Pointcloud directory not found: {args.pointcloud_dir}")
        return 1
    
    test_queries = test_text_similarity()
    
    if args.interactive:
        print("🔄 Interactive mode - you can test multiple queries")
        print("Type 'quit' or 'exit' to stop, 'help' for example queries")
        print()
        
        while True:
            try:
                query = input("Enter text query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif query.lower() == 'help':
                    print("Example queries:", ", ".join(test_queries))
                    continue
                elif not query:
                    print("⚠️  Please enter a non-empty query")
                    continue
                
                print(f"\n🔍 Searching for: '{query}'")
                run_visualization_with_text_query(
                    args.pointcloud_dir, query,
                    mode=args.mode,
                    remote_host=args.remote_host,
                    remote_port=args.remote_port,
                    top_k=args.top_k,
                    threshold=args.threshold,
                    clip_model=args.clip_model,
                    create_mesh=args.create_mesh
                )
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
    
    else:
        # Single query mode
        if args.query:
            query = args.query
        else:
            print("Available example queries:", ", ".join(test_queries))
            query = input("\nEnter your text query: ").strip()
            if not query:
                print("❌ No query provided")
                return 1
        
        print(f"\n🔍 Searching for: '{query}'")
        run_visualization_with_text_query(
            args.pointcloud_dir, query,
            mode=args.mode,
            remote_host=args.remote_host,
            remote_port=args.remote_port,
            top_k=args.top_k,
            threshold=args.threshold,
            clip_model=args.clip_model,
            create_mesh=args.create_mesh
        )
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 

