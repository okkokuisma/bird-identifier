from os import path, walk
import requests
from pydub import AudioSegment
import io
from occurrence_data import load_filtered_occurrences, parse_multimedia

def download_songs(multimedia_df, song_dir_path='/Volumes/COCO-DATA/songs/'):
    for i, (id, url, format) in enumerate(zip(multimedia_df['gbifID'], multimedia_df['identifier'], multimedia_df['format'])):
        song_path = path.join(song_dir_path, str(id))
        try:
            if not path.isfile(song_path):
                res = requests.get(url, timeout=10)

                if format == 'audio/vnd.wave':
                    raw = io.BytesIO(res.content)     
                    audio = AudioSegment.from_file(raw, format='wav')
                    audio.export(song_path, format='mp3')
                else:
                    with open(song_path, 'wb+') as f:
                        f.write(bytes(res.content))
        except:
            continue

        if (i % 100 == 0):
            print('Downloaded file [%d]/[%d]\r'%(i, multimedia_df.shape[0]), end="")


def get_downloaded_song_ids(song_dir_path='/Volumes/COCO-DATA/songs/'):
    fs = []
    for _, _, fnames in walk(song_dir_path):
        for f in fnames:
            fs.append(int(f))
    
    return fs

def get_preprocessed_song_ids(song_dir_path='/Volumes/COCO-DATA/songs_npy/'):
    fs = []
    for _, _, fnames in walk(song_dir_path):
        for f in fnames:
            fs.append(int(f.strip('.npy')))
    
    return fs

if __name__ == '__main__':
    occurrences = load_filtered_occurrences()
    multimedia_df = parse_multimedia(occurrences)
    download_songs(multimedia_df)