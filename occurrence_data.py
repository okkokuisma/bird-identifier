import pandas as pd

def parse_species(file_path='/Users/okkokuisma/projektit/birds/occurence_data/taxon-export.tsv'):
    species = pd.read_csv(file_path, sep='\t')
    species = species.dropna()
    species = species.sort_values('Havaintomäärä Suomesta', ascending=False).iloc[:260, :]
    return species

def load_occurrences(file_path='/Volumes/COCO-DATA/0000764-250426092105405/occurrence.txt'):
    return pd.read_csv(file_path, sep='\t', skiprows=[5644])

def drop_occurrence_cols(occurrences, cols_to_keep=['gbifID', 'species', 'decimalLatitude', 'decimalLongitude']):
    return occurrences.loc[:, cols_to_keep]

def filter_finnish_species(occurrences, species):
    return occurrences[occurrences['species'].isin(species['Tieteellinen nimi'])]

def get_occurrence_count_by_species(occurrences):
    return occurrences.groupby('species').count()['gbifID']

def drop_uncommon_species(occurrences):
    num_occurrences_by_species = get_occurrence_count_by_species(occurrences)
    mask = (num_occurrences_by_species < 50) 
    dropped_species = num_occurrences_by_species[mask]
    occurrences = occurrences.loc[~occurrences['species'].isin(dropped_species.index.to_list()), :]
    return occurrences

def filter_occurrences(occurrences, species):
    occurrences = filter_finnish_species(occurrences, species)
    occurrences = drop_uncommon_species(occurrences)
    return occurrences

def get_occurrences():
    species = parse_species()
    occurrences = load_occurrences()
    occurrences = drop_occurrence_cols(occurrences)
    occurrences = filter_occurrences(occurrences, species)
    occurrences['species'] = pd.Categorical(occurrences['species'])
    return occurrences

def save_filtered_data(occurrences, file_path='/Volumes/COCO-DATA/0000764-250426092105405/filtered_occurrence_data.parquet'):
    occurrences.to_parquet(file_path)

def load_filtered_occurrences(file_path='/Volumes/COCO-DATA/0000764-250426092105405/filtered_occurrence_data.parquet'):
    return pd.read_parquet(file_path)

def load_augmented_occurrences(file_path='/Volumes/COCO-DATA/0000764-250426092105405/augmented_occurrences.parquet'):
    return pd.read_parquet(file_path)

def parse_multimedia(occurences, file_path='/Volumes/COCO-DATA/0000764-250426092105405/multimedia.txt'):
    multimedia_df = pd.read_csv(file_path, sep='\t')
    multimedia_df = multimedia_df[multimedia_df['gbifID'].isin(occurences['gbifID'])]
    multimedia_df = multimedia_df[multimedia_df['type'] == 'Sound']
    multimedia_df = multimedia_df.reset_index(drop=True)
    return multimedia_df