import tarfile, json, glob, os, io
from occurrence_data import load_filtered_occurrences

occ = load_filtered_occurrences()  # with gbifID & species
occ['species'] = occ['species'].cat.codes
files = glob.glob('/Volumes/COCO-DATA/songs_npy/*.npy')
files = sorted(files)
shard_size = 1000

for i in range(0, len(files), shard_size):
    shard_id = i // shard_size
    with tarfile.open(f'/Volumes/COCO-DATA/shards/shard-{shard_id:06d}.tar', 'w') as tar:
        for fpath in files[i:i+shard_size]:
            key = os.path.basename(fpath).replace('.npy','')
            tar.add(fpath, arcname=f'{key}.npy')
            species = int(occ.loc[occ['gbifID']==int(key),'species'])
            info = json.dumps({'__key__': key, 'species': species}).encode('utf-8')
            tarinfo = tarfile.TarInfo(name=f'{key}.json')
            tarinfo.size = len(info)
            tar.addfile(tarinfo, io.BytesIO(info))
        
        if (i % 100 == 0):
            print('Processed file [%d]/[%d]\r'%(i, len(files)), end="")