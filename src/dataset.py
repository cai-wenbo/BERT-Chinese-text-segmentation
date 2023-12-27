import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertTokenizer
import csv
import re
import copy


'''
read the data from the corpus file
and extract the (text, label) pairs

labels: [PAD]:0, S:1  B:2  E:3  M:4 P:5
'''
def get_text_labels(data_path, max_length):
    buffer = 16
    text_list  = list()
    label_list = list()

    with open(data_path, 'r') as file:
        line = file.readline()

        text  = str()
        label = list()
        sentence       = str()
        sentence_label = list()

        while line:
            pattern_characters = r'\['
            line = re.sub(pattern_characters, '', line)
            word_pairs = line.split()
            if len(word_pairs) > 0:
                text = ""
                label.clear()
                sentence = ""
                sentence_label.clear()
                for word_pair in word_pairs[1:]:
                    word_pair = word_pair.split("/")
                    sentence = sentence + word_pair[0]

                    if word_pair[1] == 'w':
                    #  append the sentence to text
                        sentence_label = sentence_label + [5] * len(word_pair[0])

                    elif len(word_pair[0]) == 1:
                        sentence_label.append(1)
                    else:
                        sentence_label = sentence_label + [2] + [4] * (len(word_pair[0]) - 2) + [3]
                        

                    if word_pair[1] == 'w' or len(sentence) > max_length - buffer:
                        if len(sentence) + len(text) > max_length - 2:
                            text_list.append(text)
                            label_list.append(label.copy())
                            text  = ""
                            label.clear()
                        text  = text  + sentence
                        label = label + sentence_label
                        sentence       = ""
                        sentence_label.clear()

                
                if len(sentence) + len(text) > max_length - 2:
                    text_list.append(text)
                    label_list.append(label.copy())
                    text_list.append(sentence)
                    label_list.append(sentence_label.copy())
                else:
                    text  = text  + sentence
                    label = label + sentence_label
                    text_list.append(text)
                    label_list.append(label.copy())

            #  read the next line
            line = file.readline()
        file.close()
    return text_list, label_list




class PKCorpus(Dataset):
    def __init__(self, data_path, max_length):
        '''
        extract the text and label each symbol from the txt file
        '''
        text_list, label_list = get_text_labels(data_path, max_length)

        length_list = [len(lst) for lst in label_list]

        #  pad the labels
        label_list = [label + [0] * (max_length - len(label)) for label in label_list]


        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        encoding = tokenizer(text_list, padding = 'max_length', truncation=True, add_special_tokens=False, max_length=max_length)


        self.text_list   = encoding['input_ids']
        self.mask_list   = encoding['attention_mask']
        self.label_list  = label_list
        self.length_list = length_list
        





    def __len__(self):
        return len(self.text_list)


    def __getitem__(self, idx):
        text  = self.text_list[idx]
        label = self.label_list[idx]
        mask  = self.mask_list[idx]
        length = self.length_list[idx]

        #  tensorlize
        text_tensor   = torch.tensor(text   , dtype = torch.long)
        label_tensor  = torch.tensor(label  , dtype = torch.long)
        mask_tensor   = torch.tensor(mask   , dtype = torch.long)
        length_tensor = torch.tensor(length , dtype = torch.long)

        
        return text_tensor, label_tensor, mask_tensor, length_tensor
