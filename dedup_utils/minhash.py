# from .minhash import ...

from datasketch import MinHash
from .shingling import shingles

def build_minhash(text, num_perm, shingle_size):
    m = MinHash(num_perm=num_perm)
    for shingle in shingles(text, shingle_size):
        m.update(shingle.encode('utf8'))
    return m