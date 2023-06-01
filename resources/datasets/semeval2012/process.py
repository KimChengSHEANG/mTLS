# combine dataset, fix html chars and unknown chars

import chunk
from pathlib import Path;import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))# -- fix path --

import Levenshtein
from source.helper import * 
from source.resources import *
from source.preprocessor import *
from tqdm import tqdm
import sys 

CUR_DIR = Path(__file__).resolve().parent
FILES = [CUR_DIR / 'TEST.txt', CUR_DIR / 'TRIAL.txt']

semeval_data = []
for filepath in FILES:
    print(get_file_encoding(filepath))
    for line in yield_lines(filepath):
        semeval_data.append(line)

all_sentences = []
all_sentences.extend(load_data(Dataset.BenchLS)['text'].tolist())
all_sentences.extend(load_data(Dataset.LexMTurk)['text'].tolist())
all_sentences.extend(load_data(Dataset.NNSeval)['text'].tolist())
print(len(all_sentences))    

def sentence_similarity(sentence, sentences):
    score = 0 
    for s in sentences:
        score = Levenshtein.ratio(sentence, s)
        if score > 0.9:
            return score, s 
    return score, sentence


new_sentences = []
new_data = []
for line in tqdm(semeval_data):
    chunks = line.split('\t')
    sentence = chunks[0]
    # print(sentence) 
    # score, new_sentence = sentence_similarity(sentence, all_sentences)
    # if score > 0.9:
    #     print(score, ':', '='*80)
    #     print(sentence)
    #     print(new_sentence)
    #     new_sentence = new_sentence.capitalize()

    # new_sentences.append('\t'.join([new_sentence] + chunks[1:]))
    chunks[0] = ftfy.fix_encoding(chunks[0])
    new_sentences.append('\t'.join(chunks))
    
write_lines(new_sentences, CUR_DIR / 'semeval2012.tsv')
    