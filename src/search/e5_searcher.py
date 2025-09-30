import glob
import torch
import faiss
import numpy as np
import os

from typing import List, Dict, Tuple, Optional
from datasets import Dataset

from src.search.simple_encoder import SimpleEncoder
from src.data.data_utils import load_corpus, load_wiki18_corpus
from src.logger_config import logger

def _get_all_shards_path(index_dir: str) -> List[str]:
    path_list = glob.glob("{}/*-shard-*.pt".format(index_dir))
    assert len(path_list) > 0

    def _parse_shard_idx(p: str) -> int:
        return int(p.split("-shard-")[1].split(".")[0])

    path_list = sorted(path_list, key=lambda path: _parse_shard_idx(path))
    logger.info("Embeddings path list: {}".format(path_list))
    return path_list


class E5Searcher:

    def __init__(
        self,
        index_dir: str,
        model_name_or_path: str = "intfloat/e5-large-v2",
        verbose: bool = False,
        single_index_file: Optional[str] = None
    ):
        self.model_name_or_path = model_name_or_path
        self.index_dir = index_dir
        self.verbose = verbose
        self.single_index_file = single_index_file

        n_gpus: int = torch.cuda.device_count()
        self.gpu_ids: List[int] = list(range(n_gpus))

        self.encoder: SimpleEncoder = SimpleEncoder(
            model_name_or_path=self.model_name_or_path,
            max_length=64,
        )
        self.encoder.to(self.gpu_ids[-1])

        # Check if we should load a single FAISS index or multiple PT shards
        if self.single_index_file and os.path.exists(self.single_index_file):
            self._load_single_faiss_index()
            self.corpus = load_wiki18_corpus()
        else:
            self._load_pt_shards()
            self.corpus: Dataset = load_corpus()

    def _load_single_faiss_index(self):
        """Load a single FAISS index file"""
        logger.info(f"Loading FAISS index from {self.single_index_file}")
        
        # Load FAISS index
        self.faiss_index = faiss.read_index(self.single_index_file)
        
        # Move to GPU if available
        if len(self.gpu_ids) > 0:
            res = faiss.StandardGpuResources()
            res.useFloat16 = True
            self.faiss_index = faiss.index_cpu_to_gpu(res, self.gpu_ids[-1], self.faiss_index)
        
        logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
        self.use_faiss = True

    def _load_pt_shards(self):
        """Load multiple PyTorch embedding shards"""
        shard_paths = _get_all_shards_path(self.index_dir)
        all_embeddings: torch.Tensor = torch.cat(
            [
                torch.load(
                    p, weights_only=True, map_location=lambda storage, loc: storage
                )
                for p in shard_paths
            ],
            dim=0,
        )
        logger.info(f"Load {all_embeddings.shape[0]} embeddings from {self.index_dir}")

        split_embeddings = torch.chunk(all_embeddings, len(self.gpu_ids))
        self.embeddings: List[torch.Tensor] = [
            split_embeddings[i].to(f"cuda:{self.gpu_ids[i]}", dtype=torch.float16)
            for i in range(len(self.gpu_ids))
        ]
        self.use_faiss = False

    @torch.no_grad()
    def batch_search(self, queries: List[str], k: int, **kwargs) -> List[List[Dict]]:
        query_embed: torch.Tensor = self.encoder.encode_queries(queries)
        
        if self.use_faiss:
            return self._batch_search_faiss(query_embed, k)
        else:
            return self._batch_search_pytorch(query_embed, k)

    def _batch_search_faiss(self, query_embed: torch.Tensor, k: int) -> List[List[Dict]]:
        """Search using FAISS index"""
        # Convert to numpy and ensure correct dtype
        query_np = query_embed.cpu().numpy().astype(np.float32)
        
        # Search in FAISS index
        scores, indices = self.faiss_index.search(query_np, k)
        
        results_list: List[List[Dict]] = []
        for query_idx in range(len(query_np)):
            results: List[Dict] = []
            for score, idx in zip(scores[query_idx], indices[query_idx]):
                if idx != -1:  # FAISS returns -1 for invalid results
                    results.append(
                        {
                            "doc_id": int(idx),
                            "score": float(score),
                        }
                    )

                    if self.verbose:
                        results[-1].update(self.corpus[int(idx)])
            results_list.append(results)

        return results_list

    def _batch_search_pytorch(self, query_embed: torch.Tensor, k: int) -> List[List[Dict]]:
        """Search using PyTorch embeddings (original implementation)"""
        query_embed = query_embed.to(dtype=self.embeddings[0].dtype)
        
        batch_sorted_score, batch_sorted_indices = self._compute_topk(query_embed, k=k)

        results_list: List[List[Dict]] = []
        for query_idx in range(len(query_embed)):
            results: List[Dict] = []
            for score, idx in zip(
                batch_sorted_score[query_idx], batch_sorted_indices[query_idx]
            ):
                results.append(
                    {
                        "doc_id": int(idx.item()),
                        "score": score.item(),
                    }
                )

                if self.verbose:
                    results[-1].update(self.corpus[int(idx.item())])
            results_list.append(results)

        return results_list

    def _compute_topk(
        self, query_embed: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_score_list: List[torch.Tensor] = []
        batch_sorted_indices_list: List[torch.Tensor] = []

        idx_offset = 0
        for i in range(len(self.embeddings)):
            query_embed = query_embed.to(self.embeddings[i].device)
            score = torch.mm(query_embed, self.embeddings[i].t())
            sorted_score, sorted_indices = torch.topk(score, k=k, dim=-1, largest=True)

            sorted_indices += idx_offset
            batch_score_list.append(sorted_score.cpu())
            batch_sorted_indices_list.append(sorted_indices.cpu())
            idx_offset += self.embeddings[i].shape[0]

        batch_score = torch.cat(batch_score_list, dim=1)
        batch_sorted_indices = torch.cat(batch_sorted_indices_list, dim=1)
        # only keep the top k results based on batch_score
        batch_score, top_indices = torch.topk(batch_score, k=k, dim=-1, largest=True)
        batch_sorted_indices = torch.gather(
            batch_sorted_indices, dim=1, index=top_indices
        )

        return batch_score, batch_sorted_indices
