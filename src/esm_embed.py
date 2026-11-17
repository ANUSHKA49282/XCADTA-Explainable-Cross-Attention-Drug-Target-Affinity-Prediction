
# Extract ESM-2 protein embeddings and save them for reuse.
# Embeddings are generated once and loaded during model training and inference.

import argparse
import os
import pickle
import time

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from data_utils import sequence_hash


def load_unique_sequences(pairs_pickle: str):
    # Load protein sequences from the dataset.
    with open(pairs_pickle, 'rb') as f:
        obj = pickle.load(f)

    if not obj:
        return []

    if isinstance(obj[0], (tuple, list)):
        sequences = {item[1] for item in obj}
    else:
        sequences = set(obj)

    sequences = {
        str(sequence).strip().upper()
        for sequence in sequences
        if isinstance(sequence, str) and sequence.strip()
    }

    return sorted(sequences)


@torch.inference_mode()
def extract_embeddings(
    sequences,
    model_name: str,
    l_max: int = 1000,
    device: str = None,
    batch_size: int = 4,
    expected_dim: int = 1280
):
    if l_max <= 0:
        raise ValueError("l_max must be greater than zero")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    device = device or (
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"Loading {model_name} on {device} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    actual_dim = model.config.hidden_size

    if actual_dim != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dim}, "
            f"but the selected ESM-2 model produces {actual_dim}."
        )

    embeddings = {}
    start_time = time.time()

    for i in tqdm(
        range(0, len(sequences), batch_size),
        desc="ESM-2 embedding"
    ):
        batch_seqs = sequences[i:i + batch_size]
        truncated = [sequence[:l_max] for sequence in batch_seqs]

        tokens = tokenizer(
            truncated,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=l_max + 2
        )

        tokens = {
            key: value.to(device)
            for key, value in tokens.items()
        }

        output = model(**tokens)
        hidden = output.last_hidden_state

        for j, sequence in enumerate(batch_seqs):
            # Remove BOS and EOS tokens.
            n_real = min(len(sequence), l_max)

            embedding = hidden[
                j, 1:1 + n_real
            ].detach().cpu()

            if embedding.shape != (n_real, expected_dim):
                raise ValueError(
                    f"Unexpected embedding shape for sequence "
                    f"{sequence_hash(sequence)}: {embedding.shape}"
                )

            embeddings[sequence_hash(sequence)] = embedding

    elapsed = time.time() - start_time

    print(
        f"Done: {len(embeddings)} sequences embedded in "
        f"{elapsed / 60:.1f} min "
        f"({elapsed / max(1, len(embeddings)):.2f} s/seq)"
    )

    if elapsed > 3000:
        print(
            "WARNING: Extraction took more than 50 minutes. "
            "Consider reducing the protein subsample for future runs."
        )

    return embeddings


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--sequences_pickle',
        type=str,
        required=True,
        help='Pickle containing protein sequences or dataset triples'
    )

    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Output path for cached protein embeddings'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='facebook/esm2_t33_650M_UR50D'
    )

    parser.add_argument('--l_max', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--expected_dim', type=int, default=1280)

    args = parser.parse_args()

    if args.expected_dim <= 0:
        parser.error('--expected_dim must be greater than zero')

    sequences = load_unique_sequences(args.sequences_pickle)

    if not sequences:
        raise ValueError("No protein sequences were found in the input file")

    print(f"{len(sequences)} unique protein sequences to embed.")

    embeddings = extract_embeddings(
        sequences,
        args.model,
        l_max=args.l_max,
        batch_size=args.batch_size,
        expected_dim=args.expected_dim
    )

    output_dir = os.path.dirname(args.out)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch.save(embeddings, args.out)

    print(f"Saved cached embeddings dict to {args.out}")


if __name__ == '__main__':
    main()
