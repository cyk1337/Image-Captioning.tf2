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
@file: FeatureExtractor.py
@time: 5/5/20 10:36 PM
@desc：       
               
'''
from utils.utils import *


class BaseFeatureExtractor(object):
    def __init__(self, feat_subdir, bsz):
        self.bsz = bsz
        self.feat_subdir = feat_subdir
        os.makedirs(self.feat_subdir, exist_ok=True)

    @staticmethod
    def load_image(image_path, size=(299, 299)):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, size)
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, image_path


class InceptionV3Extractor(BaseFeatureExtractor):
    def __init__(self, cache_dir, bsz=16):
        super().__init__(cache_dir, bsz)
        img_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        input = img_model.input
        hidden_layer = img_model.layers[-1].output
        self.extractor_model = tf.keras.Model(input, hidden_layer)

    def __call__(self, image_paths):
        encode_imgs = sorted(set(image_paths))

        img_dataset = tf.data.Dataset.from_tensor_slices(encode_imgs)
        img_dataset = img_dataset.map(BaseFeatureExtractor.load_image,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(self.bsz)

        for img, path in img_dataset:
            batch_features = self.extractor_model(img)
            batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
            for bf, p in zip(batch_features, path):
                img_path = p.numpy().decode('utf-8')
                save_path = os.path.join(self.feat_subdir, os.path.basename(img_path))
                np.save(save_path, bf.numpy())


class ResNet101V2Extractor(BaseFeatureExtractor):
    def __init__(self, cache_dir, bsz=16):
        super().__init__(cache_dir, bsz)
        img_model = tf.keras.applications.ResNet101V2(include_top=False, weights='imagenet')
        input = img_model.input
        hidden_layer = img_model.layers[-1].output
        self.extractor_model = tf.keras.Model(input, hidden_layer)

    def __call__(self, image_paths):
        encode_imgs = sorted(set(image_paths))

        img_dataset = tf.data.Dataset.from_tensor_slices(encode_imgs)
        img_dataset = img_dataset.map(BaseFeatureExtractor.load_image,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(self.bsz)

        for img, path in img_dataset:
            batch_features = self.extractor_model(img)
            batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
            for bf, p in zip(batch_features, path):
                img_path = p.numpy().decode('utf-8')
                save_path = os.path.join(self.feat_subdir, os.path.basename(img_path))
                np.save(save_path, bf.numpy())


class CNNExtractor(BaseFeatureExtractor):
    def __init__(self, cache_dir, model=None, bsz=16):
        super().__init__(cache_dir, bsz)
        if model is None:
            raise NotImplementedError
        img_model = model(include_top=False, weights='imagenet')
        input = img_model.input
        hidden_layer = img_model.layers[-1].output
        self.extractor_model = tf.keras.Model(input, hidden_layer)

    def __call__(self, image_paths):
        encode_imgs = sorted(set(image_paths))

        img_dataset = tf.data.Dataset.from_tensor_slices(encode_imgs)
        img_dataset = img_dataset.map(BaseFeatureExtractor.load_image,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(self.bsz)

        for img, path in img_dataset:
            batch_features = self.extractor_model(img)
            batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
            for bf, p in zip(batch_features, path):
                img_path = p.numpy().decode('utf-8')
                save_path = os.path.join(self.feat_subdir, os.path.basename(img_path))
                np.save(save_path, bf.numpy())
