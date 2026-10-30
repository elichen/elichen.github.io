"""The engine's FLY2 connectome layout (little endian): b'FLY2', u32 n, u32 m, u32 offsets[n+1],
then for every presynaptic neuron its targets in increasing order as varint deltas (the first
one absolute), then every signed count as a zigzag varint."""
import numpy as np


def varint(a):
    a = a.astype(np.uint64)
    nbytes = np.ones(len(a), np.int64)
    x = a >> np.uint64(7)
    while (x > 0).any():
        nbytes += x > 0
        x >>= np.uint64(7)
    pos = np.concatenate([[0], np.cumsum(nbytes)[:-1]])
    buf = np.zeros(nbytes.sum(), np.uint8)
    x = a.copy()
    for k in range(nbytes.max()):
        m = nbytes > k
        more = (nbytes[m] > k + 1).astype(np.uint8) << 7
        buf[pos[m] + k] = (x[m] & np.uint64(0x7f)).astype(np.uint8) | more
        x >>= np.uint64(7)
    return buf.tobytes()


def fly2(n, pre, post, w):
    """Bytes of a FLY2 file for connections pre -> post with signed synapse counts w."""
    assert pre.max() < n and post.max() < n
    order = np.lexsort((post, pre))
    pre, post, w = pre[order], post[order], w[order].astype(np.int64)
    offsets = np.zeros(n + 1, np.uint32)
    np.cumsum(np.bincount(pre, minlength=n), out=offsets[1:])
    delta = post.astype(np.int64).copy()
    delta[1:] -= post[:-1]
    first = offsets[:-1][np.diff(offsets) > 0]
    delta[first] = post[first]
    assert (delta >= 0).all()
    zigzag = (w << 1) ^ (w >> 63)
    return b'FLY2' + np.array([n, len(pre)], np.uint32).tobytes() + offsets.tobytes() + varint(delta) + varint(zigzag)
