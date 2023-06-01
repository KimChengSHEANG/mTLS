import os
from pathlib import Path
import tarfile
import tempfile
import zipfile
import gzip, shutil
from source.constants import DATASETS_DIR, EXP_DIR, Dataset
from source.pretty_downloader import download
from source.metrics import flatten, sort_candidates_by_frequency, sort_candidates_by_ranking
from source.helper import tokenize, yield_lines
import json
import pandas as pd


def get_dataset_filepath(dataset):
    return DATASETS_DIR / f'{dataset}.tsv'


def get_tuning_log_dir():
    log_dir = EXP_DIR / 'tuning_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

def get_last_experiment_dir():
    return sorted(list(EXP_DIR.glob('exp_*')), reverse=True)[0]


def download_fasttext_embedding(lang):
    
    dest_dir = Path(tempfile.gettempdir())
    filename = f'cc.{lang}.300.vec'
    filepath = dest_dir / filename
    # if filepath.exists(): return filepath

    url = f'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/{filename}.gz'
    download_filepath = download(url, dest_dir, show_progress=True)
    print("Extracting: ", download_filepath.name)
    with gzip.open(download_filepath, 'rb') as f:
            with open(filepath, 'wb') as f_out:
                shutil.copyfileobj(f, f_out)
    download_filepath.unlink()
    return filepath 

            
def unzip(file_path, dest_dir=None):  # sourcery skip: extract-duplicate-method
    file_path = str(file_path)
    if dest_dir is None:
        dest_dir = os.path.dirname(file_path)
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
    elif file_path.endswith("tar.gz") or file_path.endswith("tgz"):
        tar = tarfile.open(file_path, "r:gz")
        tar.extractall(dest_dir)
        tar.close()
    elif file_path.endswith(".gz"):
        tofile = file_path.replace('.gz', '')
        with open(file_path, 'rb') as inf, open(tofile, 'wb') as tof:
            decom_str = gzip.decompress(inf.read())
            tof.write(decom_str)
    elif file_path.endswith("tar"):
        tar = tarfile.open(file_path, "r:")
        tar.extractall(dest_dir)
        tar.close()


def __normalize_string(text):
        text = text.replace('``', '"')
        text = text.replace('`', "'")
        text = text.replace("''", '"')
        text = text.strip('"')

        return text 

def __load_data_with_index(dataset):
        dataset_filepath = get_dataset_filepath(dataset)
        lines = yield_lines(dataset_filepath)
        docs = []

        for line in lines:
            line = line.strip().lower()
            chunks = line.split('\t')
            text = chunks[0].strip()
            text = __normalize_string(text)
            complex_word = chunks[1].strip()
            candidates = chunks[3:]
            candidates = [tuple(candidate.split(':')) for candidate in candidates]
            candidates = [(word.strip(), index) for index, word in candidates]
            
            candidates = sort_candidates_by_ranking(candidates)
            
            doc = {'text': text,
                        'complex_word': complex_word,
                        'complex_word_index': chunks[2],
                        'candidates': json.dumps(candidates)}
            docs.append(doc)
        return pd.DataFrame(docs)
    
def word_index_in_text(text, word):
    # words = text.lower().split(' ')
    words = tokenize(text.lower())
    return words.index(word.lower()) 

def __load_data(dataset):
    dataset_filepath = get_dataset_filepath(dataset)
    lines = yield_lines(dataset_filepath)
    docs = []

    for line in lines:
        line = line.strip().lower()
        chunks = line.split('\t')
        text = chunks[0].strip()
        text = __normalize_string(text)
        complex_word = chunks[1].strip()
        candidates = chunks[2:]
        candidates = [word.strip() for word in candidates] 
        candidates = sort_candidates_by_frequency(candidates)
        doc = {'text': text,
                    'complex_word': complex_word,
                    'complex_word_index': word_index_in_text(text, complex_word),
                    'candidates': json.dumps(candidates)}
        docs.append(doc)
    return pd.DataFrame(docs)

def __load_data_EASIER(dataset):
    dataset_filepath = get_dataset_filepath(dataset)
    lines = yield_lines(dataset_filepath)
    docs = []
    for line in lines:
        line = line.strip().lower()
        chunks = line.split('\t')
        text = chunks[0].strip()
        text = __normalize_string(text)
        complex_word = chunks[1].strip()
        candidates = chunks[2:]
        candidates = [[word.strip()] for word in candidates] 
        # candidates = sort_candidates_by_frequency(candidates)
        doc = {'text': text,
                    'complex_word': complex_word,
                    'complex_word_index': word_index_in_text(text, complex_word),
                    'candidates': json.dumps(candidates)}
        docs.append(doc)
    return pd.DataFrame(docs)
    
def load_dataset(dataset):
    if dataset in [Dataset.EASIER, Dataset.EASIER_GPT]:
        return __load_data_EASIER(dataset)
    elif dataset in [Dataset.BenchLS, Dataset.NNSeval]:
        return __load_data_with_index(dataset)
    else:
        return __load_data(dataset) 
    
def update_candidates(df):
    # convert candidates string array and also flatten it.
    df['candidates'] = df['candidates'].apply(json.loads) # transform it to array
    df['list_candidates'] = df['candidates'].apply(flatten) # flatten the array 2D to 1D
    return df

def split_data(data, frac=0.8, seed=42):
        data_train = data.sample(frac=frac, random_state=seed)
        data_valid = data.drop(data_train.index)
        return data_train, data_valid     
        
def split_data_train_valid_test(data, frac=0.8, seed=42):
    data_train, data_valid = split_data(data, frac, seed)
    data_valid, data_test = split_data(data_valid, frac=0.5, seed=seed)
    return data_train, data_valid, data_test

# if __name__ == '__main__':
    # print(get_temp_filepath())
