#!/usr/bin/env python3
"""
Interactive Text Search for Featurized Pointclouds using Rerun SDK

This script provides real-time interactive text search capabilities:
1. Terminal input: Type queries directly in the terminal
2. File watching: Edit query.txt file and see results update
3. Socket interface: Send queries via network socket

Usage:
  python visualize_interactive_text_search.py data/ARKitScenes_fpt --file-type 42447230

Controls:
  - Type in terminal for immediate search
  - Edit 'query.txt' file for file-based search
  - Press 'q' + Enter to quit
  - Press 'clear' + Enter to clear highlights
"""

import rerun as rr
import numpy as np
import torch
import argparse
import os
import sys
import time
import threading
import queue
import select
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import socket
import json

# Import visualization functions
from visualize_featurized_pointcloud import (
    discover_featurized_files,
    load_featurized_pointcloud,
    features_to_colors_pca,
    create_text_similarity_highlights,
    estimate_normals_and_mesh,  # Add mesh creation
    HAS_CLIP_ENCODER,
    HAS_OPEN3D  # Add open3d availability check
)

# Add the path to locate-3d for importing ClipEncoder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'locate-3d'))
try:
    from preprocessing.image_features.clip_encoder import ClipEncoder
except ImportError:
    print("❌ ClipEncoder not available. This script requires CLIP functionality.")
    sys.exit(1)

class QueryFileHandler(FileSystemEventHandler):
    """Handler for watching query file changes."""
    
    def __init__(self, query_queue, query_file):
        self.query_queue = query_queue
        self.query_file = Path(query_file).name
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        if Path(event.src_path).name == self.query_file:
            try:
                with open(event.src_path, 'r') as f:
                    query = f.read().strip()
                if query:
                    self.query_queue.put(("file", query))
                    print(f"📁 File query: '{query}'")
            except Exception as e:
                print(f"❌ Error reading query file: {e}")

class InteractiveTextSearch:
    """Main class for interactive text search visualization."""
    
    def __init__(self, files_info, file_key, clip_model_version="ViT-B/32", create_mesh=True):
        self.files_info = files_info
        self.file_key = file_key
        self.clip_model_version = clip_model_version
        self.create_mesh = create_mesh
        
        # Load data once
        print("🔄 Loading pointcloud data...")
        self.points, self.rgb, self.features_info = load_featurized_pointcloud(files_info[file_key])
        
        if 'clip' not in self.features_info:
            raise ValueError("❌ No CLIP features found in pointcloud. Text search requires CLIP features.")
        
        print(f"✅ Loaded {len(self.points)} points with CLIP features")
        
        # Initialize CLIP encoder
        print("🤖 Initializing CLIP encoder...")
        self.clip_encoder = self._initialize_clip_encoder()
        
        # Setup queues and control
        self.query_queue = queue.Queue()
        self.running = True
        self.current_query = ""
        self.last_similarities = None
        
        # Query file path
        self.query_file = Path("query.txt")
        self._setup_query_file()
        
        # Pre-compute mesh for base pointcloud if requested
        self.base_mesh_vertices = None
        self.base_mesh_faces = None
        self.base_mesh_colors = None
        if self.create_mesh and HAS_OPEN3D:
            print("🔺 Pre-computing base mesh...")
            self.base_mesh_vertices, self.base_mesh_faces, self.base_mesh_colors = estimate_normals_and_mesh(
                self.points, self.rgb, method='ball_pivoting'
            )
    
    def _initialize_clip_encoder(self):
        """Initialize CLIP encoder with auto-detection."""
        feature_dim = self.features_info['clip'].shape[1]
        
        if feature_dim == 512:
            detected_version = "ViT-B/32"
        elif feature_dim == 768:
            detected_version = "ViT-L/14"
        else:
            detected_version = self.clip_model_version
            
        print(f"🔍 Auto-detected CLIP model: {detected_version} (feature dim: {feature_dim})")
        
        try:
            clip_encoder = ClipEncoder(version=detected_version)
            print(f"✅ CLIP encoder ready on device: {clip_encoder.device}")
            return clip_encoder
        except Exception as e:
            print(f"❌ Failed to initialize CLIP encoder: {e}")
            raise
    
    def _setup_query_file(self):
        """Setup the query file for file-based input."""
        if not self.query_file.exists():
            with open(self.query_file, 'w') as f:
                f.write("chair\n")
            print(f"📝 Created query file: {self.query_file}")
        else:
            print(f"📝 Using existing query file: {self.query_file}")
    
    def start_file_watcher(self):
        """Start file watcher for query.txt."""
        event_handler = QueryFileHandler(self.query_queue, self.query_file)
        observer = Observer()
        observer.schedule(event_handler, path=str(self.query_file.parent), recursive=False)
        observer.start()
        print(f"👀 Watching {self.query_file} for changes...")
        return observer
    
    def start_terminal_input(self):
        """Start terminal input thread."""
        def terminal_input_thread():
            print("\n" + "="*60)
            print("🎯 INTERACTIVE TEXT SEARCH READY")
            print("="*60)
            print("💡 How to search:")
            print(f"   • Type queries directly here and press Enter")
            print(f"   • Edit '{self.query_file}' file in any text editor")
            print("   • Type 'clear' to remove highlights")
            print("   • Type 'q' to quit")
            print("   • Type 'help' for more commands")
            print("="*60)
            
            while self.running:
                try:
                    # Use select for non-blocking input on Unix systems
                    if hasattr(select, 'select'):
                        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                        if ready:
                            query = sys.stdin.readline().strip()
                        else:
                            continue
                    else:
                        # Fallback for Windows
                        query = input("🔍 Enter search query: ").strip()
                    
                    if query:
                        if query.lower() == 'q':
                            self.running = False
                            break
                        elif query.lower() == 'clear':
                            self.query_queue.put(("terminal", ""))
                        elif query.lower() == 'help':
                            self._show_help()
                        elif query.lower().startswith('threshold='):
                            threshold = float(query.split('=')[1])
                            self.query_queue.put(("threshold", threshold))
                        elif query.lower().startswith('topk='):
                            top_k = int(query.split('=')[1])
                            self.query_queue.put(("topk", top_k))
                        else:
                            self.query_queue.put(("terminal", query))
                            print(f"🔍 Searching for: '{query}'")
                            
                except (EOFError, KeyboardInterrupt):
                    self.running = False
                    break
                except Exception as e:
                    print(f"❌ Input error: {e}")
        
        thread = threading.Thread(target=terminal_input_thread, daemon=True)
        thread.start()
        print("⌨️  Terminal input thread started")
        return thread
    
    def _show_help(self):
        """Show help information."""
        print("\n" + "="*50)
        print("📚 INTERACTIVE SEARCH HELP")
        print("="*50)
        print("Commands:")
        print("  • <text>           - Search for text")
        print("  • clear            - Clear all highlights")
        print("  • threshold=0.25   - Set similarity threshold")
        print("  • topk=100         - Set number of results")
        print("  • help             - Show this help")
        print("  • q                - Quit")
        print("\nExamples:")
        print("  • chair")
        print("  • red sofa")
        print("  • wooden table")
        print("  • threshold=0.15")
        print("="*50 + "\n")
    
    def process_text_query(self, query, top_k=200, threshold=0.2):
        """Process a text query and return highlights."""
        if not query or not query.strip():
            # Clear highlights
            rr.log("world/text_similarity_highlights", rr.Clear())
            rr.log("world/highlighted_mesh", rr.Clear())
            
            # Clear search stats with better formatting
            rr.log("search_stats/current_query", rr.TextLog("No active search", level=rr.TextLogLevel.INFO))
            rr.log("search_stats/num_results", rr.Scalar(0))
            rr.log("search_stats/top_similarity", rr.Scalar(0.0))
            rr.log("search_stats/mean_similarity", rr.Scalar(0.0))
            rr.log("search_stats/threshold", rr.Scalar(threshold))
            return
        
        try:
            # Create highlights
            highlight_indices, highlight_points, highlight_colors, similarities = create_text_similarity_highlights(
                self.points, 
                self.features_info['clip'], 
                query,
                clip_encoder=self.clip_encoder,
                top_k=top_k,
                similarity_threshold=threshold,
                highlight_color=[1.0, 0.8, 0.0]
            )
            
            self.last_similarities = similarities
            
            if len(highlight_points) > 0:
                # Update highlights in Rerun with larger points
                rr.log("world/text_similarity_highlights", 
                       rr.Points3D(highlight_points, colors=highlight_colors, radii=0.03))
                
                # Create mesh for highlighted points if mesh creation is enabled
                if self.create_mesh and HAS_OPEN3D and len(highlight_points) > 100:
                    print(f"🔺 Creating mesh for {len(highlight_points)} highlighted points...")
                    try:
                        highlight_mesh_vertices, highlight_mesh_faces, highlight_mesh_colors = estimate_normals_and_mesh(
                            highlight_points, highlight_colors, method='ball_pivoting'
                        )
                        
                        if highlight_mesh_vertices is not None and highlight_mesh_faces is not None:
                            if highlight_mesh_colors is not None:
                                rr.log("world/highlighted_mesh", 
                                       rr.Mesh3D(vertex_positions=highlight_mesh_vertices, 
                                               triangle_indices=highlight_mesh_faces,
                                               vertex_colors=highlight_mesh_colors))
                            else:
                                rr.log("world/highlighted_mesh", 
                                       rr.Mesh3D(vertex_positions=highlight_mesh_vertices, 
                                               triangle_indices=highlight_mesh_faces))
                            print(f"✅ Highlighted mesh created: {len(highlight_mesh_vertices)} vertices, {len(highlight_mesh_faces)} faces")
                    except Exception as e:
                        print(f"⚠️  Could not create highlighted mesh: {e}")
                
                # Log comprehensive search statistics with better organization
                rr.log("search_stats/current_query", rr.TextLog(f"Query: '{query}'", level=rr.TextLogLevel.INFO))
                rr.log("search_stats/num_results", rr.Scalar(len(highlight_points)))
                rr.log("search_stats/top_similarity", rr.Scalar(float(similarities.max())))
                rr.log("search_stats/mean_similarity", rr.Scalar(float(similarities.mean())))
                rr.log("search_stats/threshold", rr.Scalar(threshold))
                rr.log("search_stats/top_k", rr.Scalar(top_k))
                
                # Log similarity distribution
                similarity_percentiles = np.percentile(similarities, [25, 50, 75, 90, 95])
                rr.log("search_stats/similarity_p25", rr.Scalar(float(similarity_percentiles[0])))
                rr.log("search_stats/similarity_median", rr.Scalar(float(similarity_percentiles[1])))
                rr.log("search_stats/similarity_p75", rr.Scalar(float(similarity_percentiles[2])))
                rr.log("search_stats/similarity_p90", rr.Scalar(float(similarity_percentiles[3])))
                rr.log("search_stats/similarity_p95", rr.Scalar(float(similarity_percentiles[4])))
                
                # Create a detailed summary log
                summary = f"""✨ SEARCH RESULTS FOR: '{query}'
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Results: {len(highlight_points):,} / {len(self.points):,} points ({len(highlight_points)/len(self.points)*100:.1f}%)
📊 Similarity: {similarities.max():.3f} (max) | {similarities.mean():.3f} (mean) | {similarity_percentiles[1]:.3f} (median)
🎛️  Settings: threshold={threshold:.2f} | top_k={top_k}
🔍 Distribution: 25%={similarity_percentiles[0]:.3f} | 75%={similarity_percentiles[2]:.3f} | 95%={similarity_percentiles[4]:.3f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
                
                rr.log("search_stats/detailed_summary", rr.TextLog(summary, level=rr.TextLogLevel.INFO))
                
                print(f"✨ Found {len(highlight_points)} matches for '{query}' (threshold: {threshold:.2f})")
                print(f"   Top similarity: {similarities.max():.3f}, Mean: {similarities.mean():.3f}")
                
            else:
                # No matches found
                rr.log("world/text_similarity_highlights", rr.Clear())
                rr.log("world/highlighted_mesh", rr.Clear())
                
                # Log no results stats
                rr.log("search_stats/current_query", rr.TextLog(f"Query: '{query}' (NO MATCHES)", level=rr.TextLogLevel.WARN))
                rr.log("search_stats/num_results", rr.Scalar(0))
                rr.log("search_stats/top_similarity", rr.Scalar(float(similarities.max()) if len(similarities) > 0 else 0.0))
                rr.log("search_stats/mean_similarity", rr.Scalar(float(similarities.mean()) if len(similarities) > 0 else 0.0))
                rr.log("search_stats/threshold", rr.Scalar(threshold))
                
                no_match_summary = f"""⚠️  NO MATCHES FOR: '{query}'
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Results: 0 / {len(self.points):,} points 
📊 Max similarity: {similarities.max():.3f} (below threshold {threshold:.2f})
💡 Try: lower threshold (threshold=0.1) or different terms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
                
                rr.log("search_stats/detailed_summary", rr.TextLog(no_match_summary, level=rr.TextLogLevel.WARN))
                
                print(f"⚠️  No matches found for '{query}' with threshold {threshold:.2f}")
                print(f"   Max similarity was {similarities.max():.3f} - try lowering threshold or using different terms")
                
        except Exception as e:
            print(f"❌ Error processing query '{query}': {e}")
            rr.log("search_stats/error", rr.TextLog(f"Error: {str(e)}", level=rr.TextLogLevel.ERROR))
    
    def run_interactive_session(self, port=9878):
        """Run the main interactive session."""
        # Initialize Rerun
        rr.init("Interactive_Text_Search", spawn=False)
        rr.serve_grpc(grpc_port=port)
        print(f"🌐 Rerun server started on port {port}")
        
        # Setup coordinate frame
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
        rr.set_time_seconds("timeline", 0.0)
        
        # Log the base pointcloud
        print("📊 Logging base pointcloud...")
        rr.log("world/pointcloud_rgb", 
               rr.Points3D(self.points, colors=self.rgb, radii=0.008), 
               static=True)
        
        # Log base mesh if available
        if self.base_mesh_vertices is not None and self.base_mesh_faces is not None:
            print("🔺 Logging base mesh...")
            if self.base_mesh_colors is not None:
                rr.log("world/base_mesh", 
                       rr.Mesh3D(vertex_positions=self.base_mesh_vertices, 
                               triangle_indices=self.base_mesh_faces,
                               vertex_colors=self.base_mesh_colors), 
                       static=True)
            else:
                rr.log("world/base_mesh", 
                       rr.Mesh3D(vertex_positions=self.base_mesh_vertices, 
                               triangle_indices=self.base_mesh_faces), 
                       static=True)
            print(f"✅ Base mesh logged: {len(self.base_mesh_vertices)} vertices, {len(self.base_mesh_faces)} faces")
        
        # Log CLIP feature visualization
        print("🎨 Logging CLIP feature visualization...")
        clip_colors = features_to_colors_pca(self.features_info['clip'], method='hsv')
        rr.log("world/pointcloud_clip_features", 
               rr.Points3D(self.points, colors=clip_colors, radii=0.008), 
               static=True)
        
        # Log comprehensive initial stats with better organization
        bbox_min = self.points.min(axis=0)
        bbox_max = self.points.max(axis=0)
        bbox_size = bbox_max - bbox_min
        
        rr.log("pointcloud_stats/total_points", rr.Scalar(len(self.points)), static=True)
        rr.log("pointcloud_stats/clip_feature_dim", rr.Scalar(self.features_info['clip'].shape[1]), static=True)
        rr.log("pointcloud_stats/bbox_size_x", rr.Scalar(float(bbox_size[0])), static=True)
        rr.log("pointcloud_stats/bbox_size_y", rr.Scalar(float(bbox_size[1])), static=True)
        rr.log("pointcloud_stats/bbox_size_z", rr.Scalar(float(bbox_size[2])), static=True)
        rr.log("pointcloud_stats/bbox_volume", rr.Scalar(float(np.prod(bbox_size))), static=True)
        
        if self.base_mesh_vertices is not None:
            rr.log("pointcloud_stats/mesh_vertices", rr.Scalar(len(self.base_mesh_vertices)), static=True)
            rr.log("pointcloud_stats/mesh_faces", rr.Scalar(len(self.base_mesh_faces)), static=True)
        
        # Create a comprehensive pointcloud summary
        mesh_info = ""
        if self.base_mesh_vertices is not None:
            mesh_info = f"""
🔺 Base Mesh: {len(self.base_mesh_vertices):,} vertices, {len(self.base_mesh_faces):,} faces"""
        
        pointcloud_summary = f"""📊 INTERACTIVE TEXT SEARCH READY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
☁️  Pointcloud: {len(self.points):,} points
🧠 Features: {self.features_info['clip'].shape[1]}D CLIP features
📦 Bounding Box: [{bbox_size[0]:.2f} × {bbox_size[1]:.2f} × {bbox_size[2]:.2f}] units{mesh_info}
🎯 Ready for interactive text search!

💡 How to search:
   • Type queries in terminal: "chair", "red table", "wooden furniture"
   • Edit query.txt file in any text editor
   • Adjust settings: "threshold=0.15", "topk=300"
   • Type "help" for more commands
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        rr.log("welcome/instructions", rr.TextLog(pointcloud_summary, level=rr.TextLogLevel.INFO), static=True)
        
        # Initialize search stats structure
        rr.log("search_stats/current_query", rr.TextLog("No active search", level=rr.TextLogLevel.INFO), static=True)
        rr.log("search_stats/num_results", rr.Scalar(0), static=True)
        rr.log("search_stats/threshold", rr.Scalar(0.2), static=True)
        rr.log("search_stats/top_k", rr.Scalar(200), static=True)
        
        # Start watchers and input
        file_observer = self.start_file_watcher()
        terminal_thread = self.start_terminal_input()
        
        # Settings
        current_threshold = 0.2
        current_top_k = 200
        
        try:
            # Process initial query from file
            if self.query_file.exists():
                with open(self.query_file, 'r') as f:
                    initial_query = f.read().strip()
                if initial_query:
                    print(f"🔍 Processing initial query: '{initial_query}'")
                    self.process_text_query(initial_query, current_top_k, current_threshold)
            
            # Main processing loop
            print("🚀 Interactive session started! Use Rerun viewer to see results.")
            
            while self.running:
                try:
                    # Check for new queries (with timeout)
                    source, query = self.query_queue.get(timeout=0.5)
                    
                    if source == "threshold":
                        current_threshold = query
                        print(f"🎛️  Updated threshold to: {current_threshold}")
                        # Re-process current query with new threshold
                        if self.current_query:
                            self.process_text_query(self.current_query, current_top_k, current_threshold)
                    
                    elif source == "topk":
                        current_top_k = query
                        print(f"🎛️  Updated top-k to: {current_top_k}")
                        # Re-process current query with new top-k
                        if self.current_query:
                            self.process_text_query(self.current_query, current_top_k, current_threshold)
                    
                    else:
                        # Regular text query
                        self.current_query = query
                        self.process_text_query(query, current_top_k, current_threshold)
                        
                        # Update query file if query came from terminal
                        if source == "terminal" and query:
                            with open(self.query_file, 'w') as f:
                                f.write(query + '\n')
                    
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️  Interactive session interrupted")
        finally:
            self.running = False
            file_observer.stop()
            file_observer.join()
            print("🛑 Interactive session ended")

def main():
    parser = argparse.ArgumentParser(description="Interactive Text Search for Featurized Pointclouds")
    parser.add_argument("pointcloud_dir", help="Directory containing featurized pointcloud .pt files")
    parser.add_argument("--file-type", default="combined", help="Which .pt file to use")
    parser.add_argument("--port", type=int, default=9878, help="Rerun server port")
    parser.add_argument("--clip-model", default="ViT-B/32", choices=["ViT-B/32", "ViT-L/14"], 
                       help="CLIP model version")
    parser.add_argument("--create-mesh", action="store_true", default=True,
                       help="Create ball_pivoting mesh for better visualization (default: True)")
    parser.add_argument("--no-mesh", action="store_true",
                       help="Disable mesh creation for faster loading")
    
    args = parser.parse_args()
    
    # Handle mesh creation flag
    create_mesh = args.create_mesh and not args.no_mesh
    
    # Discover files
    try:
        files_info = discover_featurized_files(args.pointcloud_dir)
        print("📁 Available files:")
        for key, path in files_info.items():
            print(f"   {key}: {os.path.basename(path)}")
    except Exception as e:
        print(f"❌ Error discovering files: {e}")
        return
    
    # Validate file selection
    if args.file_type not in files_info:
        print(f"❌ File type '{args.file_type}' not found.")
        print(f"Available options: {list(files_info.keys())}")
        return
    
    # Check CLIP encoder availability
    if not HAS_CLIP_ENCODER:
        print("❌ CLIP encoder not available. This script requires CLIP functionality.")
        return
    
    # Check mesh creation capability
    if create_mesh and not HAS_OPEN3D:
        print("⚠️  Open3D not available. Disabling mesh creation.")
        create_mesh = False
    
    # Start interactive session
    try:
        interactive_search = InteractiveTextSearch(
            files_info, 
            args.file_type, 
            args.clip_model,
            create_mesh=create_mesh
        )
        
        if create_mesh:
            print("🔺 Mesh creation enabled - ball_pivoting method will be used for better visualization")
        else:
            print("📊 Point cloud only mode - faster loading, no mesh")
            
        interactive_search.run_interactive_session(args.port)
    except Exception as e:
        print(f"❌ Error starting interactive session: {e}")
        raise

if __name__ == "__main__":
    main() 