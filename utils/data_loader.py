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
@file: data_loader.py
@time: 5/5/20 6:46 PM
@desc：       
               
'''
from utils.feat_extractor import *


class BaseLoader(object):
    def __init__(self, data_dir, cache_dir, feat_type, min_count=5):
        self.split_annotations = defaultdict(list)
        self.split_img_path = defaultdict(list)
        self.word_idx = {
            PAD: pad_id,
            UNK: unk_id,
            START: start_id,
            END: end_id,
        }
        self.idx_word = {v: k for k, v in self.word_idx.items()}
        self.vocab = Counter()
        self.min_count = min_count

        self.cache_dir = os.path.abspath(cache_dir)
        self.img_ft_dir = os.path.join(self.cache_dir, 'img_fts')
        self.data_dir = self.cache_dir if data_dir is None or os.path.exists(data_dir) else os.path.abspath(data_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.feat_extractors = {
            'inception_v3': InceptionV3Extractor,
            'resnet101_v2': ResNet101V2Extractor,
        }
        self.feat_type = feat_type
        self.feat_subdirs = {
            "densenet_121": os.path.join(self.img_ft_dir, 'densenet_121'),
            "inception_v3": os.path.join(self.img_ft_dir, 'inception_v3'),
            "inception_resnet_v2": os.path.join(self.img_ft_dir, 'inception_resnet_v2'),
            "resnet50": os.path.join(self.img_ft_dir, 'resnet50'),
            "resnet50_v2": os.path.join(self.img_ft_dir, 'resnet50_v2'),
            "resnet101": os.path.join(self.img_ft_dir, 'resnet101'),
            "resnet101_v2": os.path.join(self.img_ft_dir, 'resnet101_v2'),
        }
        self.cache_paths = {
            'img_fts': self.feat_subdirs[self.feat_type],
            'word_idx': os.path.join(self.cache_dir, 'word_idx.pkl'),
            'idx_word': os.path.join(self.cache_dir, 'idx_word.pkl'),
            'train': os.path.join(self.cache_dir, 'train.pkl'),
            'val': os.path.join(self.cache_dir, 'val.pkl'),
            'test': os.path.join(self.cache_dir, 'test.pkl'),
            'ref_train': os.path.join(self.cache_dir, 'ref_train.pkl'),
            'ref_val': os.path.join(self.cache_dir, 'ref_val.pkl'),
            'ref_test': os.path.join(self.cache_dir, 'ref_test.pkl'),
            'split_annotations': os.path.join(self.cache_dir, 'split_annotations.pkl')
        }

        self.cnn_extractors = {
            'densenet_121': tf.keras.applications.DenseNet121,
            'inception_v3': tf.keras.applications.InceptionV3,
            'inception_resnet_v2': tf.keras.applications.InceptionResNetV2,
            'resnet50': tf.keras.applications.ResNet50,
            'resnet50_v2': tf.keras.applications.ResNet50V2,
            'resnet101': tf.keras.applications.ResNet101,
            'resnet101_v2': tf.keras.applications.ResNet101V2,

        }

    def get_word_idx(self):
        if not os.path.exists(self.cache_paths['word_idx']):
            raise ValueError('Vocabulary file does not exist!')
        else:
            self.word_idx = self.load_pkl(self.cache_paths['word_idx'])
            self.idx_word = self.load_pkl(self.cache_paths['idx_word'])
        return self.word_idx

    def build_vocab(self):
        self.vocab = {k: v for k, v in self.vocab.items() if v >= self.min_count}
        for i, word in enumerate(sorted(self.vocab, key=self.vocab.get, reverse=True), len(self.word_idx)):
            word = word.lower()
            self.word_idx[word] = i
            self.idx_word[i] = word
        self.save_pkl(self.word_idx, self.cache_paths['word_idx'])
        self.save_pkl(self.idx_word, self.cache_paths['idx_word'])

    def extract_feats(self):
        # extractor = self.feat_extractors[self.feat_type](self.cache_paths['img_fts'])
        # for v in self.all_images_path.values():
        #     extractor(v)

        # for k in self.cnn_extractors:
        k = self.feat_type
        print(f'Extracting data from CNN:{k} ...')
        extractor = CNNExtractor(self.feat_subdirs[k], model=self.cnn_extractors[k])
        for v in self.split_img_path.values():
            extractor(v)

    def load_train_val_data(self):
        train_captions, train_imgids, train_img_path, = self.load_pkl(self.cache_paths['train'])
        train_imgs = [train_img_path[imgid] for imgid in train_imgids]

        val_img_captions, val_img_path = self.load_pkl(self.cache_paths['val'])
        val_captions = [val_img_captions[imgid][0] for imgid in val_img_path]
        val_imgs = list(val_img_path.values())
        val_img_ids = list(val_img_path.keys())
        return (train_imgids, train_imgs, train_captions), (val_img_ids, val_imgs, val_captions)

    def load_test_data(self):
        test_img_captions, test_img_path = self.load_pkl(self.cache_paths['test'])
        test_captions = [test_img_captions[imgid][0] for imgid in test_img_path]
        test_imgs = list(test_img_path.values())
        img_ids = list(test_img_path.keys())
        return img_ids, test_imgs, test_captions

    def load_val_ref(self):
        val_ref = self.load_pkl(self.cache_paths['ref_val'])
        return val_ref

    def load_test_ref(self):
        test_ref = self.load_pkl(self.cache_paths['ref_test'])
        return test_ref

    def fit_on_texts(self):
        raise NotImplementedError

    def texts_to_sequences(self, texts: list) -> list:
        raise NotImplementedError

    def tokens_to_sequences(self, tokens: list) -> list:
        if len(self.word_idx) <= 4:
            raise ValueError("Invalid vocabulary dict1")
        seq = [self.word_idx.get(w, self.word_idx[UNK]) for w in tokens]
        return seq

    def sequences_to_texts(self, sequences):
        texts_list = []
        for seq in sequences:
            ws = []
            for idx in seq:
                w = self.idx_word.get(idx, UNK)
                ws.append(w)
            texts_list.append(" ".join(ws))
        return texts_list

    def exist_cache(self):
        return True if all([os.path.exists(f) for f in self.cache_paths.values()]) else False

    @staticmethod
    def save_pkl(data, path):
        with codecs.open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"{path} saved!")

    @staticmethod
    def load_pkl(path):
        with codecs.open(path, 'rb') as f:
            data = pickle.load(f)
        return data

    def map_func(self, ids, img_name, cap):
        img_path = os.path.join(self.feat_subdirs[self.feat_type], img_name.decode('utf-8'))
        img_tensor = np.load(img_path + '.npy')
        return ids, img_tensor, cap

    def set_shapes(self, ids, img, cap, ids_shape, img_shape, cap_shape):
        ids.set_shape(ids_shape)
        img.set_shape(img_shape)
        cap.set_shape(cap_shape)
        return ids, img, cap

    def data_generator(self, img_ids, imgs, caps, bsz, buffer_size=1000, seed=RANDOM_SEED, shuffle=True):
        # imgs = tf.data.Dataset.from_tensor_slices(imgs)
        img_ids = np.array(img_ids, dtype=np.int32)
        # caps = tf.data.Dataset.from_tensor_slices(caps)
        # dataset = tf.data.Dataset.zip((imgs, caps))

        dataset = tf.data.Dataset.from_tensor_slices((img_ids, imgs, caps))
        dataset = dataset.map(
            lambda item0, item1, item2: tf.numpy_function(self.map_func, [item0, item1, item2],
                                                          [tf.int32, tf.float32, tf.int32]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            lambda ids, img, cap: self.set_shapes(ids, img, cap, img_ids.shape[1:], (None, None), caps.shape[1:]))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed).batch(bsz)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset


class MSCOCOLoader(BaseLoader):
    def __init__(self, data_dir=None, cache_dir='./data/COCO', split_file='./data/dataset_coco.json',
                 feat_type="inception_v3", reset_cache=False):
        super(MSCOCOLoader, self).__init__(data_dir, cache_dir, feat_type)
        if not reset_cache and self.exist_cache():
            logging.info(f"{cache_dir} has already been saved! \n Loading from cache file...")
            return
        logging.info(f"Start to preprocess raw data ...")
        # ================= download annoatation file =================
        # annotation_folder = os.path.join(data_dir, 'annotations/')
        # if not os.path.exists(annotation_folder):
        #     annotation_zip = tf.keras.utils.get_file('caption.zip', cache_subdir=self.cache_dir,
        #                                              origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
        #                                              extract=True)
        #     annotation_file = os.path.dirname(annotation_zip) + '/annotations/captions_train2014.json'
        #     os.remove(annotation_zip)
        # else:
        #     annotation_file = os.path.join(annotation_folder, 'captions_train2014.json')
        #
        # with open(annotation_file, 'r') as f:
        #     annotations = json.load(f)

        # =============================================================

        image_train2014 = os.path.join(data_dir, 'train2014')
        image_val2014 = os.path.join(data_dir, 'val2014')
        if not os.path.exists(image_train2014):
            image_zip = tf.keras.utils.get_file('train2014.zip',
                                                cache_subdir=data_dir,
                                                origin='http://images.cocodataset.org/zips/train2014.zip',
                                                extract=True)
            # train_path = os.path.dirname(image_zip) + image_train2014
            os.remove(image_zip)
        # else:
        #     train_path = image_train2014

        if not os.path.exists(image_val2014):
            image_zip = tf.keras.utils.get_file('val2014.zip',
                                                cache_subdir=data_dir,
                                                origin='http://images.cocodataset.org/zips/val2014.zip',
                                                extract=True)
            # val_path = os.path.dirname(image_zip) + image_val2014
            os.remove(image_zip)
        # else:
        # val_path = image_val2014

        with open(split_file, 'r') as f:
            processed_annotations = json.load(f)

        for annot in processed_annotations['images']:
            data_type = annot['split'] if annot['split'] != 'restval' else 'train'
            if data_type == 'train':
                for sent in annot['sentences']:
                    for w in sent['tokens']:
                        self.vocab[w] += 1
            self.split_annotations[data_type].append(
                [annot['imgid'], annot['filename'], [sent['tokens'] for sent in annot['sentences']]])
            self.split_img_path[data_type].append(os.path.join(data_dir, annot['filepath'], annot['filename']))
        self.build_vocab()
        self.process_annotations_coco(self.split_annotations)
        self.extract_feats()

        # for k in self.all_captions:
        #     self.all_captions[k], self.all_images_path[k] = shuffle(self.all_captions[k], self.all_images_path[k],
        #                                                             random_state=RANDOM_SEED)

        # ================= load from raw =================
        # all_captions = []
        # all_img_path = []
        #
        # for annot in annotations['annotations']:
        #     caption = '<start>' + annot['caption'] + '<end>'
        #     image_id = annot['image_id']
        #     full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)
        #     all_img_path.append(full_coco_image_path)
        #     all_captions.append(caption)
        # ==================================================

    def process_annotations_coco(self, split_annotations):
        split_imgid = defaultdict(list)
        split_captions = defaultdict(list)
        dict_img_path = defaultdict(dict)
        dict_img_captions = defaultdict(dict)
        for k in split_annotations:
            imgs, caps = [], []
            # for sample in split_annotations[k]:
            #     for tokens in sample[-1]:
            #         seqs = self.tokens_to_sequences([START] + tokens + [END])
            #         imgs.append(sample[1])
            #         caps.append(seqs)
            #         # text_dict.update({'seqs': seqs})
            # if k == 'train':
            #
            # else:
            #     for sample in split_annotations[k]:
            #         seqs = [self.tokens_to_sequences([START] + tokens + [END]) for tokens in sample[-1]]
            #         caps.append(seqs)
            #         imgs.append(sample[1])

            for imgid, img_path, refs in split_annotations[k]:
                seqs = [self.tokens_to_sequences([START] + ref + [END]) for ref in refs]
                if k == 'train':
                    split_captions[k].extend(seqs)
                    split_imgid[k].extend([imgid for _ in range(len(refs))])
                else:
                    dict_img_captions[k].update({imgid: seqs})
                dict_img_path[k].update({imgid: img_path})
            if k == 'train':
                self.save_pkl([split_captions[k], split_imgid[k], dict_img_path[k]], self.cache_paths[k])
            else:
                self.save_pkl([dict_img_captions[k], dict_img_path[k]], self.cache_paths[k])
            self.save_pkl(dict_img_captions[k], self.cache_paths[f'ref_{k}'])
        self.save_pkl(split_annotations, self.cache_paths['split_annotations'])


if __name__ == '__main__':
    data_loaders = {
        "COCO": MSCOCOLoader,
    }
    data_dirs = {
        "COCO": "/home/c/Cpt/COCO",
    }
    data_name = 'COCO'
    ft_type = 'inception_v3'
    dataset = data_loaders[data_name](data_dir=data_dirs[data_name], feat_type=ft_type, reset_cache=True)
