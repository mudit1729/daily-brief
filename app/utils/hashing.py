import hashlib
import struct


def _shingle(text, n=3):
    """Generate word n-grams (shingles) from text."""
    words = text.lower().split()
    if len(words) < n:
        return [' '.join(words)] if words else []
    return [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]


def _hash64(token):
    """Produce a 64-bit hash from a string token."""
    digest = hashlib.md5(token.encode('utf-8')).digest()
    return struct.unpack('<Q', digest[:8])[0]


def _to_signed64(val):
    """Convert unsigned 64-bit int to signed (for PostgreSQL BIGINT)."""
    if val >= (1 << 63):
        val -= (1 << 64)
    return val


def simhash(text, hashbits=64):
    """
    Compute 64-bit SimHash of text.
    1. Tokenize into 3-gram shingles
    2. Hash each shingle
    3. Build weighted bit vector
    4. Return signed 64-bit integer (compatible with PostgreSQL BIGINT)
    """
    tokens = _shingle(text, n=3)
    if not tokens:
        return 0

    v = [0] * hashbits
    for token in tokens:
        h = _hash64(token)
        for i in range(hashbits):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1

    fingerprint = 0
    for i in range(hashbits):
        if v[i] > 0:
            fingerprint |= (1 << i)
    return _to_signed64(fingerprint)


def hamming_distance(a, b):
    """Count differing bits between two 64-bit integers (signed or unsigned)."""
    # Mask to 64 bits to handle signed negative values correctly
    return bin((a ^ b) & 0xFFFFFFFFFFFFFFFF).count('1')
