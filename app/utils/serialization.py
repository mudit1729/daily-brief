import numpy as np


def embedding_to_bytes(vec):
    """Convert numpy array to raw bytes for DB storage."""
    return vec.astype(np.float32).tobytes()


def bytes_to_embedding(data, dim=1536):
    """Convert raw bytes back to numpy array."""
    return np.frombuffer(data, dtype=np.float32).reshape(dim,)
