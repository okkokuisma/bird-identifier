import tarfile, json, glob, os, io
from occurrence_data import load_augmented_occurrences, load_filtered_occurrences
import numpy as np
import pandas as pd

def train_val_test_split_indices(n_values, splits=[0.7, 0.15, 0.15]):
    first_split = int(np.floor(splits[0] * n_values))
    second_split = int(first_split + splits[1] * n_values)
    indices = np.arange(n_values)
    np.random.shuffle(indices)
    train, val, test = np.split(indices, [first_split, second_split])
    return train, val, test

def save_shards(files, dir_path, occurrence_data, shard_size=1000):
    for i in range(0, len(files), shard_size):
        shard_id = i // shard_size
        shard_path = os.path.join(dir_path, f'shard-{shard_id:06d}.tar')
        with tarfile.open(shard_path, 'w') as tar:
            for fpath in files[i:i+shard_size]:
                key = os.path.basename(fpath).replace('.npy', '')
                tar.add(fpath, arcname=f'{key}.npy')
                species = int(occurrence_data.loc[occurrence_data['gbifID'] == int(key), 'species_codes'].iloc[0])
                info = json.dumps({'__key__': key, 'species': species}).encode('utf-8')
                tarinfo = tarfile.TarInfo(name=f'{key}.json')
                tarinfo.size = len(info)
                tar.addfile(tarinfo, io.BytesIO(info))
            
        if (i % 100 == 0):
            print('Processed file [%d]/[%d]\r'%(i, len(files)), end="")


if __name__ == '__main__':
    occ = load_filtered_occurrences()
    aug = load_augmented_occurrences()
    occ = pd.concat([occ, aug])
    ids = occ['gbifID']
    files = []
    for id in ids:
        files.append('/Volumes/COCO-DATA/songs_npy/' + str(id) + '.npy')
    files = np.array(files)
    train_ind, val_ind, test_ind = train_val_test_split_indices(len(files))

    save_shards(files = files[val_ind], dir_path='/Volumes/COCO-DATA/val_shards/', occurrence_data=occ)
    save_shards(files = files[test_ind], dir_path='/Volumes/COCO-DATA/test_shards/', occurrence_data=occ)
    save_shards(files = files[train_ind], dir_path='/Volumes/COCO-DATA/train_shards/', occurrence_data=occ)