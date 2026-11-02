#!/usr/bin/env python3
"""Which BANC skeletons the page draws. Run in fly-brain/: python tools/pick_skeletons.py <banc_neurons.parquet>
Writes list.json (the bucket listing), urls.txt and picked.json (cell index -> SWC file) for export_skeletons.py:
every named cell that has a skeleton, plus a seeded sample per class."""
import sys,urllib.request,json,urllib.parse,re,collections,pandas as pd,numpy as np
A="https://storage.googleapis.com/storage/v1/b/lee-lab_brain-and-nerve-cord-fly-connectome/o"
tok=None;out=[]
while True:
    q={'prefix':'compiled_data/banc_888/banc_banc_space_swc/','maxResults':1000,'fields':'items(name,size),nextPageToken'}
    if tok:q['pageToken']=tok
    r=json.load(urllib.request.urlopen(A+'?'+urllib.parse.urlencode(q)))
    out+=[(i['name'].split('/')[-1],int(i['size'])) for i in r.get('items',[])]
    tok=r.get('nextPageToken')
    if not tok:break
json.dump(out,open('list.json','w'))
print(len(out),collections.Counter(re.sub(r'^\d+','',n) for n,_ in out))
J=json.load(open('model/neurons.json'))
d=pd.read_parquet(sys.argv[1]);d['root_id']=d.root_id.astype(str)
L=json.load(open('list.json'));has={}
for n,s in L:
    k=re.match(r'\d+',n).group(0)
    if k not in has or n.endswith('_skeleton.swc'): has[k]=(n,s)   # the detailed one if both
d['file']=d.root_id.map(lambda r:has.get(r,(None,0))[0]);d['size']=d.root_id.map(lambda r:has.get(r,(None,0))[1])
rng=np.random.default_rng(1);keep=set()
def flat(v):
    if isinstance(v,dict):
        for x in v.values(): yield from flat(x)
    elif isinstance(v,list):
        for x in v:
            if isinstance(x,int): yield x
for s in (J['life'],J['named'],J['wing'],J['looms']): keep|=set(flat(s))
N=dict(descending=1155,motor=786,ascending=500,central_brain_intrinsic=2500,optic_lobe_intrinsic=2000,visual_projection=1000,ventral_nerve_cord_intrinsic=1000,sensory=500,visual_centrifugal=100,sensory_ascending=50,visceral_circulatory=50)
for k,n in N.items():
    i=np.where((d.super_class==k)&d.file.notna()&(d['size']>20000))[0];keep|=set(rng.choice(i,min(n,len(i)),replace=False).tolist())
K=sorted(i for i in keep if isinstance(d.file.iloc[i],str));print(len(keep),len(K),d['size'].iloc[K].sum()/1e9,'GB')
open('urls.txt','w').write('\n'.join(d.file.iloc[K]))
json.dump({int(i):d.file.iloc[i] for i in K},open('picked.json','w'))
