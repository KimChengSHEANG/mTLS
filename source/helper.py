# # -- fix path --
# from pathlib import Path
# import sys

# sys.path.append(str(Path(__file__).resolve().parent.parent))
# # -- end fix path --

import tempfile
from collections import Counter
from source.constants import DUMPS_DIR, EXP_DIR
import pickle
import re
import sys
import time
from contextlib import contextmanager
from functools import lru_cache, wraps
from pathlib import Path
import hashlib
import magic
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import torch 

import json
from sacremoses import MosesDetokenizer, MosesTokenizer

@lru_cache(maxsize=3)
def get_tokenizer(lang='en'):
    return MosesTokenizer(lang)

@lru_cache(maxsize=1)
def get_detokenizer(lang='en'):
    return MosesDetokenizer(lang)

def tokenize(sentence):
    return get_tokenizer().tokenize(sentence)


def get_file_encoding(filepath):
    blob = open(filepath, 'rb').read()
    m = magic.Magic(mime_encoding=True)
    return m.from_buffer(blob)

def write_lines(lines, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as fout:
        for line in lines:
            fout.write(line + '\n')


def read_lines(filepath):
    return [line.rstrip() for line in yield_lines(filepath)]


def yield_lines(filepath):
    filepath = Path(filepath)
    with filepath.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # line = ftfy.fix_text(line)
            yield line 

def count_line(filepath):
    filepath = Path(filepath)
    line_count = 0
    with filepath.open("r", encoding=get_file_encoding(filepath)) as f:
        for _ in f:
            line_count += 1
    return line_count


def get_temp_filepath(create=False):
    temp_filepath = Path(tempfile.mkstemp()[1])
    if not create:
        temp_filepath.unlink()
    return temp_filepath

def get_experiment_dir():
    dir_name = f'{int(time.time() * 1000000)}'
    path = EXP_DIR / f'exp_{dir_name}'
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_dump(filepath):
    return pickle.load(open(filepath, 'rb'))


def dump(obj, filepath):
    pickle.dump(obj, open(filepath, 'wb'))


def print_execution_time(func):
    @wraps(func)  # preserve name and doc of the function
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Execution time({func.__name__}):{time.time() - start}")
        return result

    return wrapper


def generate_hash(data):
    h = hashlib.new('md5')
    h.update(str(data).encode())
    return h.hexdigest()


@contextmanager
def log_stdout(filepath, mute_stdout=False):
    '''Context manager to write both to stdout and to a file'''

    class MultipleStreamsWriter:
        def __init__(self, streams):
            self.streams = streams

        def write(self, message):
            for stream in self.streams:
                stream.write(message)

        def flush(self):
            for stream in self.streams:
                stream.flush()

    save_stdout = sys.stdout
    log_file = open(filepath, 'w')
    if mute_stdout:
        sys.stdout = MultipleStreamsWriter([log_file])  # Write to file only
    else:
        sys.stdout = MultipleStreamsWriter([save_stdout, log_file])  # Write to both stdout and file
    try:
        yield
    finally:
        sys.stdout = save_stdout
        log_file.close()


def log_params(filepath, kwargs):
    filepath = Path(filepath)
    kwargs_str = {key: str(kwargs[key]) for key in kwargs}
    json.dump(kwargs_str, filepath.open('w'), indent=4)



def apply_line_method_to_file(line_method, input_filepath):
    output_filepath = get_temp_filepath()
    with open(input_filepath, 'r') as input_file, open(output_filepath, 'w') as output_file:
        for line in input_file:
            transformed_line = line_method(line.rstrip('\n'))
            if transformed_line is not None:
                output_file.write(transformed_line + '\n')
    return output_filepath


def lowercase_file(filepath):
    return apply_line_method_to_file(lambda line: line.lower(), filepath)


def multiprocess(func, iterable, size, desc='Processing'):
    pool = mp.Pool(mp.cpu_count())
    return list(tqdm(pool.imap(func, iterable), total=size, desc=desc))

def unique(items):
    return list(dict.fromkeys(items))

def safe_division(a, b):
    return a / b if b else 0

def save_preprocessor(preprocessor):
    DUMPS_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSOR_DUMP_FILE = DUMPS_DIR / 'preprocessor.pk'
    dump(preprocessor, PREPROCESSOR_DUMP_FILE)


def load_preprocessor():
    PREPROCESSOR_DUMP_FILE = DUMPS_DIR / 'preprocessor.pk'
    if PREPROCESSOR_DUMP_FILE.exists():
        return load_dump(PREPROCESSOR_DUMP_FILE)
    else:
        return None
    
def get_device():
    type = 'cpu'
    if torch.cuda.is_available():
        type = 'cuda'
    elif torch.backends.mps.is_available():
        type = 'mps'
    return torch.device(type)