from data_download import download_songs
from occurrence_data import load_filtered_occurrences, parse_multimedia

occurrences = load_filtered_occurrences()
multimedia_df = parse_multimedia(occurrences)
download_songs(multimedia_df)