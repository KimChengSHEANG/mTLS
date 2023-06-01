# # -- fix path --
import gc
from pathlib import Path
import sys
from tabnanny import verbose
from source.metrics import flatten

sys.path.append(str(Path(__file__).resolve().parent.parent))
# # -- end fix path --

from source.resources import download_fasttext_embedding, load_dataset, split_data
from source.constants import CACHE_DIR, DUMPS_DIR, PROCESSED_DATA_DIR, Language
from nltk.corpus import stopwords
from source import helper, wandb_config
from source.helper import get_tokenizer, tokenize
from functools import lru_cache
from string import punctuation
import numpy as np
import nltk
from tqdm import tqdm
import pandas as pd
from hyphen import Hyphenator
import re
import json 
from sentence_transformers import SentenceTransformer, util
import torch 

from joblib import Memory
from diskcache import Cache

memory = Memory(CACHE_DIR, verbose=0)
   
nltk.download('stopwords')
nltk.download('punkt')

def round(val):
    return f'{val:.2f}'


def safe_division(a, b):
    return a / b if b else 0

@lru_cache(maxsize=5)
def get_stopwords(lang):
    if lang == 'es':
        return set(stopwords.words('spanish'))
    elif lang == 'fr':
        return set(stopwords.words('french'))
    elif lang == 'en':
        return set(stopwords.words('english'))
    elif lang == 'pt':
        return set(stopwords.words('portuguese')) 
    elif lang == 'de':
        return set(stopwords.words('german'))
    else:
        return None
    
    
@lru_cache(maxsize=1024)
def is_punctuation(word):
    return not ''.join([char for char in word if char not in punctuation])


@lru_cache(maxsize=128)
def remove_punctuation(text):
    return ' '.join([word for word in tokenize(text) if not is_punctuation(word)])


def remove_stopwords(text, lang):
    stopwords = get_stopwords(lang)
    return ' '.join([w for w in tokenize(text) if w.lower() not in stopwords])



@lru_cache(maxsize=1)
def get_word2rank(lang, vocab_size=np.inf):
    filename = f'cc.{lang}.300.bin'
    model_filepath = DUMPS_DIR / f"{filename}.pk"
    if model_filepath.exists():
        return helper.load_dump(model_filepath)
    print("Preprocessing word2rank...")
    word_embeddings_filepath = download_fasttext_embedding(lang)
    lines_generator = helper.yield_lines(word_embeddings_filepath)
    word2rank = {}
    # next(lines_generator)
    for i, line in enumerate(lines_generator):
        if i >= vocab_size:
            break
        word = line.split(' ')[0]
        word2rank[word] = i
    helper.dump(word2rank, model_filepath)
    return word2rank


def download_requirements(lang):
    get_word2rank(lang)

def embed_token_source(text, complex_word):
    pattern = re.compile(rf'\b{complex_word}\b', re.IGNORECASE)
    return pattern.sub(f'[T] {complex_word} [/T]', text, count=1)

    # pattern = re.compile(rf'{source_word}', re.IGNORECASE)
    # return pattern.sub(f'[T] {source_word} [/T]', text, count=1)
    # print(text, ':\t', complex_word)
    # return text.replace(complex_word, f'[T] {complex_word} [/T]')


def embed_token_target(simple_word):
    return f'[T] {simple_word} [/T]'
    

class RatioFeature:
    def __init__(self, name, feature_extractor, target_ratio=0.8, lang='en'):
        self.lang = lang
        self.name = name
        self.feature_extractor = feature_extractor
        self.target_ratio = f'{target_ratio:.2f}'
        
    def get_target_ratio(self):
        return f'<{self.name}_{self.target_ratio}>'
        # return f'{self.target_ratio}'
   
    def extract_ratio(self, text, complex_word, simple_word, grouped_candidates):
        return f'<{self.name}_{self.feature_extractor(text, complex_word, simple_word, grouped_candidates)}>'
        # return f'{self.feature_extractor(text, complex_word, simple_word, grouped_candidates)}'


class WordLength(RatioFeature):
    def __init__(self, *args, **kwargs):
        super().__init__('WL', self.get_char_length_ratio, *args, **kwargs)

    def get_char_length_ratio(self, text, complex_word, simple_word, grouped_candidates):
        return round(safe_division(len(simple_word), len(complex_word)))


class WordRank(RatioFeature):
    def __init__(self, *args, **kwargs):
        super().__init__('WR', self.get_word_rank_ratio, *args, **kwargs)

    def get_word_rank_ratio(self, text, complex_word, simple_word, grouped_candidates):
        return round(safe_division(self.get_lexical_complexity_score(simple_word), self.get_lexical_complexity_score(complex_word)))
        # return round(safe_division(self.get_lexical_complexity_score(complex_word), self.get_lexical_complexity_score(simple_word)))

    def get_lexical_complexity_score(self, word):
        return self.get_normalized_rank(word)

    @lru_cache(maxsize=10000)
    def get_normalized_rank(self, word):
        max = len(get_word2rank(self.lang))
        rank = get_word2rank(self.lang).get(word, max)
        return np.log(1 + rank) / np.log(1 + max)
        # return np.log(1 + rank)

class WordSyllable(RatioFeature):
    def __init__(self, *args, **kwargs):
        super().__init__('WS', self.get_word_syllable_ratio, *args, **kwargs)

    @lru_cache(maxsize=1)
    def get_hypernator(self):
        if self.lang == 'en':
            return Hyphenator('en_US')
        elif self.lang == 'es':
            return Hyphenator('es_ES')
        elif self.lang == 'fr':
            return Hyphenator('fr_FR')
        elif self.lang == 'pt':
            return Hyphenator('pt_BR')
        else: 
            return None

    @lru_cache(maxsize=10**6)
    def count_syllable(self, word):
        h = self.get_hypernator()  
        return len(h.syllables(word))

    def get_word_syllable_ratio(self, text, complex_word, simple_word, grouped_candidates):
        return round(safe_division(self.count_syllable(simple_word), self.count_syllable(complex_word)))

class CandidateRanking(RatioFeature):
    def __init__(self, *args, **kwargs):
        super().__init__('CR', self.get_ranking_ratio, *args, **kwargs)
    
    def get_ranking_ratio(self, text, complex_word, simple_word, grouped_candidates):
        ranks = {0: 1.00, 1: 0.75, 2: 0.50, 3: 0.25, 4: 0.10}
        index = 4
        for i, candidates in enumerate(grouped_candidates):
            if simple_word in candidates:
                index = i
                break
        index = min(index, 4)
        return round(ranks[index])
    
    def get_ranking_ratio2(self, text, complex_word, simple_word, grouped_candidates):
        for i, candidates in enumerate(grouped_candidates, start=1):
            if simple_word in candidates:
                index = i
                break
        return round(1 - np.log10(index))   


def __normalize(val, val_min, val_max):
        return safe_division((val - val_min),(val_max - val_min))
    
@lru_cache(maxsize=1)
def __get_sentence_transformer_model():
    return SentenceTransformer('multi-qa-mpnet-base-dot-v1',  device=helper.get_device())

@memory.cache()
def __get_sentence_similarity(sent1, sent2):
    model = __get_sentence_transformer_model()
    sent1_embedding = model.encode(sent1, convert_to_tensor=True)
    sent2_embedding = model.encode(sent2, convert_to_tensor=True)
    return util.cos_sim(sent1_embedding, sent2_embedding).cpu().item() * 100

@memory.cache()
def get_similarity_scores(text, complex_word, grouped_candidates):
    grouped_candidates = json.loads(grouped_candidates) # lru_cache cannot hash array
    candidates = flatten(grouped_candidates)
    scores = {}
    for candidate in candidates: 
        text2 = text.replace(complex_word, candidate)
        scores[candidate] = __get_sentence_similarity(text, text2)
    
    val_min = min(scores.values())
    val_max = max(scores.values())
    # print(scores)
    for key, val in scores.items():
        scores[key] = __normalize(val, val_min, val_max)
        
    return scores
    
class SentenceSimilarity(RatioFeature):
    def __init__(self, *args, **kwargs):
        super().__init__('SS', self.get_similarity_score, *args, **kwargs)
        
    def get_similarity_score(self, text, complex_word, simple_word, grouped_candidates):
        grouped_candidates = json.dumps(grouped_candidates) # lru_cache cannot hash array
        scores = get_similarity_scores(text, complex_word, grouped_candidates)
        return round(scores[simple_word])
        
        
class Preprocessor:
    def __init__(self, features_kwargs, lang):
        super().__init__()
        self.lang = lang 
        # self.window_size = 10 
        # self.window_size = 15
        self.window_size = 200 
        # self.window_size = 0
        
        self.features = self.__get_features(features_kwargs)
        if features_kwargs:
            self.hash = helper.generate_hash(str(features_kwargs).encode())
            self.num_feature = len(features_kwargs)
        else:
            self.hash = "no_feature"
            self.num_feature = 0
            
        
    def __get_class(self, class_name, *args, **kwargs):
        return globals()[class_name](*args, **kwargs, lang=self.lang)

    def __get_features(self, feature_kwargs):
        return [self.__get_class(feature_name, **kwargs) for feature_name, kwargs in feature_kwargs.items()]
    
    def get_hash(self):
        return self.hash
    
    def decode_sentence(self, encoded_sentence):
        for feature in self.features:
            decoded_sentence = feature.decode_sentence(encoded_sentence)
        return decoded_sentence

    def extract_ratios(self, text, complex_word, simple_word, grouped_candidates):
        if not self.features:
            return ''
        ratios = ''
        for feature in self.features:
            val = feature.extract_ratio(text, complex_word, simple_word, grouped_candidates)
            ratios += f'{val} '
        return ratios.strip()
    
    
    def encode_sentence(self, text, complex_word):
        text = embed_token_source(text, complex_word)
        return f'{self.get_target_ratios()} {text}'
    
    
    def get_target_ratios(self):
        if not self.features:
            return ''
        ratios = ' '.join(feature.get_target_ratio() for feature in self.features)
        return ratios.rstrip()
    
    
    def crop_text(self, text, complex_word, window_size):
        return text
        # tokenizer = get_tokenizer(self.lang)
        # tokens = tokenizer.tokenize(text)
        # size = len(tokens)
        # if size <= (2*window_size): #short sentence, no need to crop
        #     return text
        # index = tokens.index(complex_word)
        # left_tokens = tokens[:index][-window_size:]
        # right_tokens = tokens[index+1:][:window_size]
        # ll = len(left_tokens)
        # lr = len(right_tokens)
        # if ll>lr:
        #     size_remaining = ll - lr 
        #     left_tokens = tokens[:index][-(window_size+size_remaining):]
        # elif lr>ll: 
        #     size_remaining = lr - ll
        #     right_tokens = tokens[index+1:][:window_size+size_remaining]
            
        
        # return ' '.join(left_tokens + [complex_word] + right_tokens) 
    
       
    def preprocess_train_set(self, data, use_mask_pred_candidates):
        download_requirements(self.lang)

        processed_data= []
        size = len(data)
        for i in tqdm(range(size), total=size):
            row = data.iloc[i]
            text = row['text']
            complex_word = row['complex_word']
            cropped_text = self.crop_text(text, complex_word, self.window_size)

            grouped_candidates = json.loads(row['candidates'])
            list_candidates = flatten(grouped_candidates)
            for simple_word in list_candidates:
                ratios = self.extract_ratios(text, complex_word, simple_word, grouped_candidates)

                source_sent = embed_token_source(cropped_text, complex_word)
                target_sent = embed_token_target(simple_word)
                if use_mask_pred_candidates:
                    joint_candidates = '\t'.join(json.loads(row['mask_pred_candidates']))    
                    source = f'simplify {self.lang}: {ratios} {source_sent} </s> {complex_word} : {joint_candidates}'
                else: 
                    source = f'simplify {self.lang}: {ratios} {source_sent}'
                item = {'source': source, 
                        'target': target_sent,
                        'complex_word': complex_word,
                        'candidates': row['candidates']}
                processed_data.append(item)

        return pd.DataFrame(processed_data)
    
    def preprocess_valid_or_test_set(self, data, use_mask_pred_candidates):
        sources = []
        size = len(data)
        for i in tqdm(range(size), total=size):
            row = data.iloc[i]
            text = row['text']
            complex_word = row['complex_word']
            cropped_text = self.crop_text(text, complex_word, self.window_size)
            
            if use_mask_pred_candidates:
                joint_candidates = '\t'.join(json.loads(row['mask_pred_candidates']))
                source = f'simplify {self.lang}: {self.encode_sentence(cropped_text, complex_word)} </s> {complex_word} : {joint_candidates}'
            else: 
                source = f'simplify {self.lang}: {self.encode_sentence(cropped_text, complex_word)}'
            sources.append(source)
            
        data['source'] = sources
        return data