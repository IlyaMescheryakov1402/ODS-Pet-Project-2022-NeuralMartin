from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import glob
import re

#
def generate_phrase(text, model, dataset_tokenizer, size='small'):
    """
    size = small, middle, big
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dict_of_size = {'extrasmall':50, 'small': 100, 'middle': 150, 'big': 200}
    ids = dataset_tokenizer.encode(text, return_tensors='pt').to(device)
    greedy_output = model.generate(ids, do_sample=True, max_length=dict_of_size[size], top_k=40, top_p=0.95)
    result = dataset_tokenizer.decode(greedy_output[0], skip_special_tokens=True)
    return result

def generate_paragraph(model, dataset_tokenizer):
    predict = generate_phrase(dataset_tokenizer.bos_token, model, dataset_tokenizer, 'extrasmall')
    for _ in range(10):
        predict += generate_phrase(predict[-1], model, dataset_tokenizer, 'extrasmall')

    return predict
