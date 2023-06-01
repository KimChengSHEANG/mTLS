from pathlib import Path    

REPO_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = REPO_DIR / 'resources'
EXP_DIR = REPO_DIR / 'experiments'
DATASETS_DIR = RESOURCES_DIR / 'datasets'
PROCESSED_DATA_DIR = RESOURCES_DIR / "processed_data"
DUMPS_DIR = RESOURCES_DIR / "dumps"
DUMPS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = RESOURCES_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class Phase:
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'
    
class Dataset:
    LexMTurk = 'lex.mturk'
    NNSeval = 'NNSeval'
    BenchLS = 'BenchLS'
    SemEval2012 = 'semeval2012'
    TSAR_EN = 'tsar-en'
    TSAR_ES = 'tsar-es'
    TSAR_PT = 'tsar-pt'
    TSAR_EN_TESTSET = 'tsar_en_test'
    ALEXSIS = 'ALEXSIS_v1.0'
    EASIER = 'EASIER500'
    EASIER_GPT = 'EASIER-gpt'
    
class Language:
    EN = 'en'
    ES = 'es'
    PT = 'pt'
    FR = 'fr'
    DE = 'de'
    
    DICT = {'en':'english', 
            'es':'spanish', 
            'pt':'portuguese', 
            'fr':'french',
            'de': 'german'}
        

class Train_Type:
    DIFF = 0 # Train and Test different data
    WHOLE = 1  # Train and Test with the same data
