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
@file: models.py
@time: 5/10/20 10:01 PM
@desc：       
               
'''
from models.base import *


class ShowAttendTell(TrainTemplate):
    """ Show Attend and Tell (ICML 2015)"""

    def __init__(self, model_config, optimizer):
        super().__init__(model_config, optimizer)
        self.encoder = CNNEncoder(model_config.embed_dim)
        self.decoder = RNNDecoder(model_config.embed_dim, model_config.h_dim, model_config.vocab_size)

    def __repr__(self):
        return "NIC model"


class DistributedShowAttendTell(DistributeTrain):
    """ Distributed baseline"""

    def __init__(self, model_config, optimizer, strategy):
        super().__init__(model_config, optimizer, strategy)
        self.encoder = CNNEncoder(model_config.embed_dim)
        self.decoder = RNNDecoder(model_config.embed_dim, model_config.h_dim, model_config.vocab_size)

    def __repr__(self):
        return "Distributed NIC model"
