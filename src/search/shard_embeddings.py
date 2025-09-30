import faiss
import torch
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import argparse

def convert_faiss_to_pytorch_shards(
    faiss_index_path: str,
    output_dir: str,
    num_shards: int = 32,
    shard_prefix: str = "embeddings-shard"
):
    """
    Convert a large FAISS index to PyTorch embedding shards like e5-large.
    
    Args:
        faiss_index_path: Path to the FAISS index file
        output_dir: Directory to save PyTorch shards
        num_shards: Number of shards to create (default: 32)
        shard_prefix: Prefix for shard files (default: "embeddings-shard")
    """
    
    print(f"Loading FAISS index from {faiss_index_path}...")
    
    # Load FAISS index on CPU
    index = faiss.read_index(faiss_index_path)
    total_vectors = index.ntotal
    dimension = index.d
    
    print(f"Index info:")
    print(f"  Total vectors: {total_vectors:,}")
    print(f"  Dimensions: {dimension}")
    print(f"  Index size: {os.path.getsize(faiss_index_path) / (1024**3):.1f} GB")
    
    # Calculate vectors per shard
    vectors_per_shard = (total_vectors + num_shards - 1) // num_shards
    print(f"  Creating {num_shards} shards with ~{vectors_per_shard:,} vectors each")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract and save embeddings in shards
    for shard_idx in range(num_shards):
        start_idx = shard_idx * vectors_per_shard
        end_idx = min((shard_idx + 1) * vectors_per_shard, total_vectors)
        current_shard_size = end_idx - start_idx
        
        if current_shard_size == 0:
            break
            
        print(f"\nProcessing shard {shard_idx + 1}/{num_shards}")
        print(f"  Vectors: {start_idx:,} to {end_idx:,} ({current_shard_size:,} vectors)")
        
        # Extract embeddings for this shard
        print("  Extracting embeddings from FAISS index...")
        shard_embeddings = []
        
        # Extract in batches to avoid memory issues
        batch_size = 10000
        for batch_start in tqdm(range(start_idx, end_idx, batch_size), 
                               desc="  Extracting batches"):
            batch_end = min(batch_start + batch_size, end_idx)
            batch_vectors = np.array([
                index.reconstruct(i) for i in range(batch_start, batch_end)
            ])
            shard_embeddings.append(batch_vectors)
        
        # Concatenate all batches for this shard
        shard_embeddings = np.concatenate(shard_embeddings, axis=0)
        
        # Convert to PyTorch tensor with float16 (like e5-large)
        print("  Converting to PyTorch tensor...")
        shard_tensor = torch.from_numpy(shard_embeddings).to(torch.float16)
        
        # Save shard
        shard_path = os.path.join(output_dir, f"{shard_prefix}-{shard_idx}.pt")
        print(f"  Saving to {shard_path}...")
        torch.save(shard_tensor, shard_path)
        
        # Verify shard
        shard_size_mb = os.path.getsize(shard_path) / (1024**2)
        print(f"Saved shard {shard_idx}: {shard_tensor.shape} ({shard_size_mb:.1f} MB)")
        
        # Free memory
        del shard_embeddings, shard_tensor
    
    print(f"\nConversion complete!")
    print(f"Output directory: {output_dir}")
    print(f"Created {shard_idx + 1} shards")
    
    # Verify total size
    total_size_gb = sum(
        os.path.getsize(os.path.join(output_dir, f)) 
        for f in os.listdir(output_dir) 
        if f.endswith('.pt')
    ) / (1024**3)
    print(f"Total size: {total_size_gb:.1f} GB")
    
    return output_dir

def verify_shards(shard_dir: str, shard_prefix: str = "embeddings-shard"):
    """Verify that shards can be loaded and have correct shapes."""
    print(f"\nVerifying shards in {shard_dir}...")
    
    shard_files = sorted([
        f for f in os.listdir(shard_dir) 
        if f.startswith(shard_prefix) and f.endswith('.pt')
    ])
    
    total_vectors = 0
    for i, shard_file in enumerate(shard_files):
        shard_path = os.path.join(shard_dir, shard_file)
        try:
            shard = torch.load(shard_path, map_location='cpu')
            total_vectors += shard.shape[0]
            print(f"  Shard {i}: {shard.shape} ✅")
        except Exception as e:
            print(f"  Shard {i}: Error loading: {e}")
    
    print(f"Verification complete: {total_vectors:,} total vectors")

def test_e5_searcher_compatibility(shard_dir: str):
    """Test that the shards work with E5Searcher."""
    print(f"\nTesting E5Searcher compatibility...")
    
    try:
        # This would use your existing E5Searcher
        from src.search.e5_searcher import E5Searcher
        
        searcher = E5Searcher(
            index_dir=shard_dir,
            model_name_or_path="intfloat/e5-base-v2"
        )
        
        # Test search
        results = searcher.batch_search(["test query"], k=5)
        print(f"E5Searcher test passed: {len(results[0])} results")
        
    except Exception as e:
        print(f"E5Searcher test failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FAISS index to PyTorch shards")
    parser.add_argument(
        "--faiss_index", 
        required=True,
        help="Path to FAISS index file"
    )
    parser.add_argument(
        "--output_dir", 
        required=True,
        help="Output directory for PyTorch shards"
    )
    parser.add_argument(
        "--num_shards", 
        type=int, 
        default=32,
        help="Number of shards to create (default: 32)"
    )
    parser.add_argument(
        "--shard_prefix", 
        default="embeddings-shard",
        help="Prefix for shard files (default: embeddings-shard)"
    )
    parser.add_argument(
        "--verify", 
        action="store_true",
        help="Verify shards after creation"
    )
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Test E5Searcher compatibility"
    )
    
    args = parser.parse_args()
    
    # Convert FAISS to PyTorch shards
    output_dir = convert_faiss_to_pytorch_shards(
        faiss_index_path=args.faiss_index,
        output_dir=args.output_dir,
        num_shards=args.num_shards,
        shard_prefix=args.shard_prefix
    )
    
    if args.verify:
        verify_shards(output_dir, args.shard_prefix)
    
    if args.test:
        test_e5_searcher_compatibility(output_dir)
    
    print(f"Index directory: {output_dir}")
    print(f"E5Searcher command:")
    print(f"E5Searcher(index_dir='{output_dir}', model_name_or_path='intfloat/e5-base-v2')")