import os
import pandas as pd
import requests

def parse_species():
    with open('/Volumes/COCO-DATA/0000764-250426092105405/bird_species.txt') as f:
        rows = []
        row = []
        for l in f:
            if not (l.startswith('Lahko') or l.startswith('Heimo') or l.strip() == 'C' or l.strip() == 'B') and l.strip():
                row.append(l.strip())
                if (len(row) == 5):
                    rows.append(row)
                    row = []

    species = pd.DataFrame(rows, columns=['abb', 'scientific', 'finnish', 'swedish', 'english'])
    species = species.loc[:, ['scientific', 'finnish', 'english']]
    return species

def get_occurences(species):
    d = pd.read_csv('/Volumes/COCO-DATA/0000764-250426092105405/occurrence.txt', sep='\t', skiprows=[5644])
    cols = [ 'gbifID', 'species', 'decimalLatitude', 'decimalLongitude' ]
    d = d.loc[:, cols]
    occurences = d[d['species'].isin(species['scientific'])]
    return occurences

def parse_multimedia(occurences):
    multimedia_df = pd.read_csv('/Volumes/COCO-DATA/0000764-250426092105405/multimedia.txt', sep='\t')
    multimedia_df = multimedia_df[multimedia_df['gbifID'].isin(occurences['gbifID'])]
    multimedia_df = multimedia_df[multimedia_df['format'] == 'audio/mpeg']
    multimedia_df = multimedia_df.reset_index(drop=True)
    return multimedia_df

def download_songs(multimedia_df):
    for id, url in zip(multimedia_df['gbifID'].values, multimedia_df['identifier'].values):
        path = f"/Volumes/COCO-DATA/songs/{id}"
        try:
            if not os.path.isfile(path):
                res = requests.get(url)

                with open(path, 'wb+') as f:
                    f.write(bytes(res.content))
        except:
            continue

species = parse_species()
occurences = get_occurences(species)
multimedia_df = parse_multimedia(occurences)
download_songs(multimedia_df.loc[:99999, :])