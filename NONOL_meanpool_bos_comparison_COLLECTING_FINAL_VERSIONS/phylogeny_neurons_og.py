import pandas as pd
import numpy as np
import torch
import argparse
import random
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from skbio import Protein
from skbio.alignment import global_pairwise_align_protein
import itertools
import json


class NeuronDistanceAnalyzer:
    """
    Compute average pairwise sequence distance (Poisson-corrected)
    among activating proteins (top-x + random-z) for each neuron.
    In this version, saves full pairwise distances for each neuron.
    """

    def __init__(self, tsv_path: str, activations_path: str,
                 output_dir: str = "neuron_distances",
                 normalize_activations: bool = True,
                 x: int = 20, z: int = 20,
                 neuron_indices: Optional[List[int]] = None,
                 random_seed: int = 42):
        self.tsv_path = tsv_path
        self.activations_path = activations_path
        self.normalize_activations = normalize_activations
        self.neuron_indices = neuron_indices
        self.x = x
        self.z = z
        self.random_seed = random_seed

        random.seed(random_seed)
        np.random.seed(random_seed)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"run_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------ Data loading ------------------

    def load_data(self):
        print("Loading data...")
        self.tsv_data = pd.read_csv(self.tsv_path, sep='\t', low_memory=False)

        loaded_data = torch.load(self.activations_path, map_location='cpu')
        if isinstance(loaded_data, dict):
            if 'embeddings' in loaded_data:
                self.activations = loaded_data['embeddings']
            else:
                tensor_keys = [k for k, v in loaded_data.items() if isinstance(v, torch.Tensor)]
                self.activations = loaded_data[tensor_keys[0]]
        elif isinstance(loaded_data, torch.Tensor):
            self.activations = loaded_data
        else:
            raise ValueError(f"Unexpected activations format: {type(loaded_data)}")

        if isinstance(self.activations, torch.Tensor):
            self.activations = self.activations.numpy()

        if self.normalize_activations:
            self._normalize_activations()

        print(f"✅ Loaded activations with shape {self.activations.shape}")

    def _normalize_activations(self):
        print("Normalizing activations to [0,10] per neuron...")
        n_proteins, n_neurons = self.activations.shape
        norm = np.zeros_like(self.activations)
        for j in range(n_neurons):
            a = self.activations[:, j]
            minv, maxv = np.min(a), np.max(a)
            norm[:, j] = a if minv == maxv else 10 * (a - minv) / (maxv - minv)
        self.activations = norm

    # ------------------ Selection ------------------

    def select_proteins_for_neuron(self, neuron_idx: int) -> List[int]:
        """Return indices of top-x + random-z activating proteins."""
        neuron_acts = self.activations[:, neuron_idx]
        sorted_idx = np.argsort(neuron_acts)[::-1]

        top_x = sorted_idx[:self.x].tolist()
        activating = np.where(neuron_acts > 0)[0].tolist()
        activating = [i for i in activating if i not in top_x]

        rand_z = random.sample(activating, min(self.z, len(activating)))
        all_idx = top_x + rand_z
        random.shuffle(all_idx)
        return all_idx

    # ------------------ Distance computation ------------------

    def pairwise_distances(self, seqs: List[str]):
        """Return list of all pairwise Poisson distances using scikit-bio, skipping invalid sequences."""
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")  # standard 20 amino acids
        clean_seqs = [s for s in seqs if set(s).issubset(valid_aas)]
        if len(clean_seqs) < 2:
            return [], np.nan

        distances = []
        index_pairs = []
        for (i, j) in itertools.combinations(range(len(clean_seqs)), 2):
            try:
                s1, s2 = Protein(clean_seqs[i]), Protein(clean_seqs[j])
                aln, _, _ = global_pairwise_align_protein(s1, s2)
                aligned1, aligned2 = str(aln[0]), str(aln[1])
                diffs = sum(a != b for a, b in zip(aligned1, aligned2))
                p = diffs / len(aligned1)
                d = np.inf if p >= 1.0 else -np.log(1 - p)
                distances.append(d)
                index_pairs.append((i, j))
            except Exception:
                continue

        avg = np.nanmean(distances) if distances else np.nan
        return distances, avg

    # ------------------ Main loop ------------------

    def run(self):
        self.load_data()

        total_neurons = self.activations.shape[1]
        selected = [i for i in self.neuron_indices if 0 <= i < total_neurons]
        print(f"Processing {len(selected)} specified neurons: {selected}")

        results = []
        out_csv = self.output_dir / "neuron_pairwise_distances.csv"
        pairwise_jsonl = self.output_dir / "neuron_pairwise_distances_per_neuron.jsonl"

        with open(pairwise_jsonl, "w") as f_jsonl:
            for idx, neuron_idx in enumerate(selected, 1):
                print(f"\n Neuron {neuron_idx}")
                idxs = self.select_proteins_for_neuron(neuron_idx)
                seqs = [s for s in self.tsv_data.loc[idxs, "Sequence"].dropna().tolist()
                        if isinstance(s, str) and len(s) > 10]

                if len(seqs) < 2:
                    print(f"  Skipping (only {len(seqs)} valid sequences)")
                    continue

                distances, avg_dist = self.pairwise_distances(seqs)
                results.append({
                    "neuron_idx": neuron_idx,
                    "num_sequences": len(seqs),
                    "avg_pairwise_distance": avg_dist,
                    "num_pairwise": len(distances)
                })
                print(f"  ✓ Avg distance = {avg_dist:.4f} (n={len(seqs)})")

                # Save distances for this neuron as JSON line
                json_line = {
                    "neuron_idx": neuron_idx,
                    "pairwise_distances": distances,
                    "num_sequences": len(seqs)
                }
                f_jsonl.write(json.dumps(json_line) + "\n")

                # Save after every 5 neurons
                if idx % 5 == 0 or idx == len(selected):
                    df = pd.DataFrame(results)
                    df.to_csv(out_csv, index=False)
                    print(f"  💾 Progress saved to {out_csv} ({idx}/{len(selected)} neurons complete)")
                    print(f"  💾 Pairwise distances saved to {pairwise_jsonl}")

        print(f"\n✅ Final results saved to {out_csv}")
        print(f"✅ Full pairwise per-neuron distances saved to {pairwise_jsonl}")


# ------------------ CLI ------------------

def main():
    parser = argparse.ArgumentParser(description="Compute average pairwise protein distances for specified neurons")
    parser.add_argument("-t", "--tsv_file", required=True)
    parser.add_argument("-a", "--activations", required=True)
    parser.add_argument("--neuron_file", required=True,
                        help="Path to a text file with one neuron index per line")
    parser.add_argument("-x", type=int, default=20)
    parser.add_argument("-z", type=int, default=20)
    parser.add_argument("-o", "--output_dir", default="neuron_distances")
    parser.add_argument("--no_normalize_activations", dest="normalize_activations", action="store_false")
    parser.set_defaults(normalize_activations=True)
    args = parser.parse_args()

    # read neuron list
    with open(args.neuron_file, "r") as f:
        neuron_indices = [int(line.strip()) for line in f if line.strip().isdigit()]
    print(f"Loaded {len(neuron_indices)} neuron indices from {args.neuron_file}")

    analyzer = NeuronDistanceAnalyzer(
        tsv_path=args.tsv_file,
        activations_path=args.activations,
        output_dir=args.output_dir,
        normalize_activations=args.normalize_activations,
        x=args.x,
        z=args.z,
        neuron_indices=neuron_indices
    )
    analyzer.run()


if __name__ == "__main__":
    main()

