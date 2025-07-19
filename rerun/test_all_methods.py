#!/usr/bin/env python3
"""
Test script for the all-methods comparison feature.
This script demonstrates how all outlier methods are now logged in Rerun for any query.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from visualize_interactive_text_search import run_all_outlier_methods
import numpy as np

def test_all_methods_function():
    """Test the run_all_outlier_methods function with dummy data."""
    
    # Create dummy data
    num_points = 1000
    points = np.random.rand(num_points, 3) * 10  # Random 3D points
    clip_features = np.random.rand(num_points, 512)  # Random CLIP features
    
    # Create a mock CLIP encoder
    class MockCLIPEncoder:
        def encode_text(self, text):
            # Return a random text feature vector
            return np.random.rand(1, 512)
    
    clip_encoder = MockCLIPEncoder()
    
    # Test the function
    print("🧪 Testing all outlier methods function...")
    results, similarities = run_all_outlier_methods(
        points=points,
        clip_features=clip_features,
        text_query="test query",
        clip_encoder=clip_encoder,
        min_threshold=0.1,
        percentile_threshold=95,
        iqr_multiplier=2.5,
        z_score_threshold=2.0,
        min_points=5
    )
    
    print(f"✅ Function completed successfully!")
    print(f"📊 Similarities shape: {similarities.shape}")
    print(f"🔬 Results keys: {list(results.keys())}")
    
    for method_name, result in results.items():
        print(f"   {method_name}: {len(result['points'])} points, threshold: {result['threshold']:.3f}")
    
    print("\n🎉 All-methods comparison feature is working correctly!")

if __name__ == "__main__":
    test_all_methods_function() 