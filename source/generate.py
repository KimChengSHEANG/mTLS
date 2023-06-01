from timeit import repeat
import torch
import re
from source.helper import unique 
from source import helper 

def extract_substitute(text):
    matches = re.findall(r'\[T\](.*?)\[\/T\]', text) # (.*?) match the first occurance 
    return matches[0] if matches else ''

@torch.no_grad()
def generate(source, model, tokenizer, max_seq_length):

    encoding = tokenizer(source, 
                         truncation=True,
                         max_length=max_seq_length,
                         padding='max_length',
                         return_tensors="pt")

    device = helper.get_device()
    input_ids = encoding["input_ids"].to(device)
    attention_masks = encoding["attention_mask"].to(device)

    beam_outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        do_sample=False,
        max_length=max_seq_length,
        num_beams=15,
        top_k=50,
        top_p=1.0, # 0.98
        repetition_penalty=2.0,
        temperature=1.0, #0.9
        early_stopping=True,
        num_return_sequences=15,
    )

    pred_sents = []
    pred_candidates = []
    for output in beam_outputs:
        sent = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        substitute = extract_substitute(sent).strip()
        # pred_sents.append(f'[{substitute}]\t{sent}')
        # sent = sent.strip()

        pred_sents.append(sent)
        pred_candidates.append(substitute)

    pred_candidates = unique(pred_candidates)
    pred_candidates = [candidate for candidate in pred_candidates if candidate]
    return pred_sents, pred_candidates
    