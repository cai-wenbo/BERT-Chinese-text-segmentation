import torch
import torch.nn as nn
from src.model import BertTagger
from src.utils import TextCoder
from transformers import BertTokenizer
import os
import argparse


def insert_char(string, char, loc):
    return string[:loc] + char + string[loc:]


def segment_text(test_config):
    model_name = "bert-base-chinese"
    text_coder = TextCoder(model_name)



    '''
    load model
    '''
    model = BertTagger.from_pretrained(test_config['model_path_src'])

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


    while True:
        src = input("enter the src word:\n")
        text_list = tokenizer.tokenize(src)
        text_tensor, mask_tensor = text_coder(src)
        text_tensor = text_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)
        with torch.no_grad():
            outputs = model(
                    text_tensor,
                    attention_mask = mask_tensor
                    )
            b_predicts = torch.argmax(outputs, dim=2)
            b_predicts = b_predicts.squeeze(0).tolist()

            for i in range(len(b_predicts)):
                loc = len(b_predicts) - i - 1
                if (b_predicts[loc] == 0) or (b_predicts[loc] == 1) or (b_predicts[loc] == 4):
                    text_list = insert_char(text_list, ['|'], loc)
            print(''.join(text_list))
                                                




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size"       , type=int   , help="vocab size"                                        , default=21128)
    parser.add_argument("--embedding_dim"    , type=int   , help="embedding dimmention"                              , default=512)
    parser.add_argument("--LSTM_hidden_size" , type=int   , help="hidden_size of the BiLSTM model"                   , default=256)
    parser.add_argument("--LSTM_num_layers"  , type=int   , help="num_layers of the BiLSTM model"                    , default=1)
    parser.add_argument("--num_labels"       , type=int   , help="types of labels"                                   , default=6)
    parser.add_argument("--sequence_length"  , type=int   , help="sequence_length"                                   , default=128)
    parser.add_argument("--model_path_src"   , type=str   , help="the directory to load model"                       , default='./saved_models/')

    
    args = parser.parse_args()

    test_config = dict()
    for arg in vars(args):
        test_config[arg] = getattr(args, arg)

    segment_text(test_config)
