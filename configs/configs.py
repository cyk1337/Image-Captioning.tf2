#!/usr/bin/env python

# -*- encoding: utf-8

'''
_____.___._______________  __.____ __________    _________   ___ ___    _____  .___ 
\__  |   |\_   _____/    |/ _|    |   \      \   \_   ___ \ /   |   \  /  _  \ |   |
 /   |   | |    __)_|      < |    |   /   |   \  /    \  \//    ~    \/  /_\  \|   |
 \____   | |        \    |  \|    |  /    |    \ \     \___\    Y    /    |    \   |
 / ______|/_______  /____|__ \______/\____|__  /  \______  /\___|_  /\____|__  /___|
 \/               \/        \/               \/          \/       \/         \/     
 

@author: Yekun Chai
@license: CASIA
@email: chaiyekun@gmail.com
@file: configs.py
@time: 5/6/20 9:56 PM
@desc：       
               
'''


class BaseConfig(object):
    def __init__(self, vocab_size, embed_dim, h_dim, maxlen, vocab_dict, idx_word, num_epochs, use_pretrained_embed,
                 embed_path, model_repr):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.maxlen = maxlen
        self.h_dim = h_dim
        self.vocab_dict = vocab_dict
        self.idx_word = idx_word
        self.num_epochs = num_epochs
        self.model_repr = model_repr
        self.embed_path = embed_path
        self.use_pretrained_embed = use_pretrained_embed
        if use_pretrained_embed and embed_path is None:
            raise ValueError("Not given pretrained embedding file path!")

    def __repr__(self):
        return self.model_repr


class ShowAttendTellConfig(BaseConfig):
    def __init__(self, vocab_size, embed_dim, h_dim, maxlen, vocab_dict, idx_word, num_epochs, use_pretrained_embed,
                 embed_path, model_repr):
        super().__init__(vocab_size, embed_dim, h_dim, maxlen, vocab_dict, idx_word, num_epochs, use_pretrained_embed,
                         embed_path, model_repr)
