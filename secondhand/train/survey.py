import pickle, numpy as np, os, collections
root='data/BRUSH'
texts=[]; npts=[]; heights=[]; widths=[]; nstrokes=[]; chars=collections.Counter(); spac=[]
for w in sorted([d for d in os.listdir(root) if d.isdigit()], key=int):
    for f in os.listdir(f'{root}/{w}'):
        if not f.isdigit(): continue
        if '_' in f: continue
        t, p, c = pickle.load(open(f'{root}/{w}/{f}','rb'))
        p=np.asarray(p); texts.append((int(w),f,t)); npts.append(len(p))
        heights.append(np.ptp(p[:,1])); widths.append(np.ptp(p[:,0])); nstrokes.append(p[:,2].sum())
        d=np.hypot(*np.diff(p[:,:2],axis=0).T); spac.append(np.median(d))
        chars.update(t)
print('samples',len(texts),'writers',len(set(w for w,_,_ in texts)))
for name,a in [('npts',npts),('height',heights),('width',widths),('strokes',nstrokes),('spacing',spac),('textlen',[len(t) for _,_,t in texts])]:
    a=np.array(a); print(name, np.percentile(a,[1,10,50,90,99]).round(1))
print(''.join(sorted(chars)), len(chars))
import random; random.seed(0)
for x in random.sample(texts,25): print(x)
c=collections.Counter(t for _,_,t in texts); print(c.most_common(10))
