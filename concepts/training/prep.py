# Decode ImageNet parquet shards into one uint8 array: short side resized to 256, center crop 256
# (a further center 224 crop is the standard eval view).
#   python prep.py        validation shards in val/   -> imgs256.npy, labels.npy
#   python prep.py train  train shards in train/      -> train256.npy, train_labels.npy
import glob, io, sys
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = '/home/aif_eng/play/sae'
S = 256
SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'val'
IMGS, LABELS = ('imgs256.npy', 'labels.npy') if SPLIT == 'val' else (f'{SPLIT}256.npy', f'{SPLIT}_labels.npy')


def load(data):
    im = Image.open(io.BytesIO(data)).convert('RGB')
    w, h = im.size
    s = S / min(w, h)
    im = im.resize((max(S, round(w * s)), max(S, round(h * s))), Image.BILINEAR)
    w, h = im.size
    l, t = (w - S) // 2, (h - S) // 2
    return np.asarray(im.crop((l, t, l + S, t + S)))


if __name__ == '__main__':
    shards = sorted(glob.glob(f'{ROOT}/{SPLIT}/*.parquet'))
    n = sum(pq.ParquetFile(f).metadata.num_rows for f in shards)
    imgs = np.lib.format.open_memmap(f'{ROOT}/{IMGS}', 'w+', np.uint8, (n, S, S, 3))
    labels = np.zeros(n, np.int64)
    i = 0
    with Pool(14) as pool:
        for f in shards:
            t = pq.read_table(f, columns=['image', 'label'])
            data = [d['bytes'] for d in t.column('image').to_pylist()]
            for j, a in enumerate(pool.imap(load, data, chunksize=32)):
                imgs[i + j] = a
            labels[i:i + len(data)] = t.column('label').to_numpy()
            i += len(data)
            print(f, i, flush=True)
    imgs.flush()
    np.save(f'{ROOT}/{LABELS}', labels)
    print('done', n, np.bincount(labels, minlength=1000).min(), np.bincount(labels, minlength=1000).max())
