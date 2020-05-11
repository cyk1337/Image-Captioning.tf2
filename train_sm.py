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
@file: main.py
@time: 5/5/20 6:45 PM
@desc：       
               
'''

from utils.data_loader import *
from models.model import *
from configs.configs import *

FLAGS = flags.FLAGS


def img_cap_flags():
    """ task to run (required to change) """
    # data
    flags.DEFINE_string("dataset", 'COCO', "['COCO', 'Flickr8k', ...]")
    flags.DEFINE_string("raw_data_path", '/home/c/Cpt/COCO', "unzipped data path")
    flags.DEFINE_boolean("reset_cache", False, "Delete processed file if exists")
    flags.DEFINE_boolean('enable_function', True, 'Enable Function?')

    flags.DEFINE_boolean("debug", True, "DEBUG")

    # model settings
    flags.DEFINE_enum('model_name', 'ShowAttendTell', ['ShowAttendTell'], 'Model name')
    flags.DEFINE_string('cnn_type', 'inception_v3', 'CNN encoder type: ')
    flags.DEFINE_integer('feat_shape', 2048, 'Feature shape')
    flags.DEFINE_boolean("use_pretrained_embed", False, "Whether to use pretrained embeddings")
    flags.DEFINE_string('embed_path', None, 'Pre-trained embedding path')
    flags.DEFINE_enum('optim_name', 'Adam', ['Adam', 'RMSProp', 'SGD'], 'Optimizer')

    flags.DEFINE_boolean("do_train", True, "Train mode")
    flags.DEFINE_boolean("do_test", True, "Inference on the test set")
    flags.DEFINE_boolean("do_predict", False, "Inference on the online prediction")
    flags.DEFINE_integer("random_seed", 2020, "fix random seed")

    # hyper-params
    flags.DEFINE_integer("max_seq_len", 30, "Max sequence length for each inputs")
    flags.DEFINE_integer("train_bsz", 100, "Batch size in the train mode")
    flags.DEFINE_integer("val_bsz", 100, "Batch size in the eval mode")
    flags.DEFINE_integer("test_bsz", 100, "Batch size in the test mode")
    flags.DEFINE_integer("max_num_words", None, "Maximum vocabulary size")
    flags.DEFINE_integer("embed_dim", 300, "Embedding dim")
    flags.DEFINE_integer("h_dim", 512, "Attn_dim")

    flags.DEFINE_float("dropout_rate", .2, "Dropout_rate")
    flags.DEFINE_boolean("do_clip", True, "Whether to clip grads")
    flags.DEFINE_float("clipnorm", 5., "Clip norm")
    flags.DEFINE_float("lr", 1e-3, "Initial learning rate")
    flags.DEFINE_integer("num_epochs", 100, "Epochs to train")
    flags.DEFINE_integer("tol", 30, "Tolerance for early stopping")
    flags.DEFINE_integer("val_every", 200, "Every K epoch to evaluate once on eval set")

    # devices
    flags.DEFINE_string("CUDA_VISIBLE_DEVICES", '3', 'specified gpu num for training')

    # model storage
    flags.DEFINE_boolean("save_models", True, "Whether to save models")
    flags.DEFINE_string("restore_dir", None, 'set path if use `do_predict` from ckpt path')

    # display
    flags.DEFINE_integer("print_every", 100, "Print train and eval every $k$ steps")


def load_dataset(dataset, raw_data_path, cnn_type, reset_cache):
    data_loaders = {
        "COCO": MSCOCOLoader,
    }
    if dataset not in data_loaders:
        raise ValueError(f"Invalid task name: {dataset}!")
    data_loader = data_loaders[FLAGS.dataset](raw_data_path, feat_type=cnn_type,
                                              reset_cache=reset_cache)
    return data_loader


def create_dataset(dataset, raw_data_path, cnn_type, reset_cache, max_seq_len, train_bsz, val_bsz,
                   debug):
    data_loader = load_dataset(dataset, raw_data_path, cnn_type, reset_cache)
    vocab_dict = data_loader.get_word_idx()
    ref_val = data_loader.load_val_ref()
    (train_imgids, train_imgs, train_caps), (val_imgids, val_imgs, val_caps) = data_loader.load_train_val_data()
    # ====================================
    if debug:
        train_imgs = train_imgs[:3000]
        train_caps = train_caps[:3000]
        train_imgids = train_imgids[:3000]
        print("\x1b[1;33;m Start debugging with 3000 data samples .. \x1b[0m")
        # val_imgids = val_imgids[:300]
        # val_imgs = val_imgs[:300]
        # val_caps = val_caps[:300]
        # print("\x1b[1;33;m Start debugging with 300 val data samples .. \x1b[0m")
    # ====================================
    # train
    padded_train_caps = tf.keras.preprocessing.sequence.pad_sequences(train_caps, padding='post', maxlen=max_seq_len)
    train_data = data_loader.data_generator(train_imgids, train_imgs, padded_train_caps, train_bsz)
    # val
    padded_val_caps = tf.keras.preprocessing.sequence.pad_sequences(val_caps, padding='post', maxlen=max_seq_len)
    val_data = data_loader.data_generator(val_imgids, val_imgs, padded_val_caps, val_bsz)
    return train_data, val_data, vocab_dict, data_loader.idx_word, ref_val


def load_test_data(dataset, raw_data_path, cnn_type, reset_cache, max_seq_len, test_bsz, debug):
    data_loader = load_dataset(dataset, raw_data_path, cnn_type, reset_cache)
    vocab_dict = data_loader.get_word_idx()
    test_refs = data_loader.load_test_ref()
    (test_imgids, test_imgs, test_caps) = data_loader.load_test_data()
    # ====================================
    # if debug:
    #     test_imgids = test_imgids[:300]
    #     test_imgs = test_imgs[:300]
    #     test_caps = test_caps[:300]
    #     print("Start debugging with 300 test data samples ..")
    # ====================================
    # test
    padded_test_caps = tf.keras.preprocessing.sequence.pad_sequences(test_caps, padding='post', maxlen=max_seq_len)
    test_data = data_loader.data_generator(test_imgids, test_imgs, padded_test_caps, test_bsz)
    return test_data, vocab_dict, data_loader.idx_word, test_refs


def create_model(model_name, model_config, optimizer):
    model_collections = {
        'ShowAttendTell': ShowAttendTell,
    }
    if model_name not in model_collections:
        raise ValueError(f"Invalid model_name {model_name}!")
    model = model_collections[model_name]
    model = model(model_config, optimizer)
    logging.info(f"{'>' * 30} run model: \x1b[1;33;m\t{model_name}\t \x1b[0m {'>' * 30}")
    return model


def create_model_config(model_name, vocab_size, vocab_dict, idx_word, num_epochs, model_repr):
    base_params = (
        vocab_size, FLAGS.embed_dim, FLAGS.h_dim, FLAGS.max_seq_len, vocab_dict, idx_word, num_epochs,
        FLAGS.use_pretrained_embed, FLAGS.embed_path, model_repr,
    )
    config_collections = {
        'ShowAttendTell': (ShowAttendTellConfig, base_params),
        # ...
    }

    config_params = config_collections[model_name]
    if len(config_params) != 2:
        raise ValueError("invalid config_collections: (`ConfigClass`, params) ")
    model_config = config_params[0](*config_params[1])
    return model_config


def create_optimizer(optim_name, lr, clipnorm):
    optimizers = {
        'adam': tf.keras.optimizers.Adam,
        'rmsprop': tf.keras.optimizers.RMSprop,
        'sgd': tf.keras.optimizers.SGD,
    }
    optimizer = optimizers[optim_name.lower()](lr, clipnorm=clipnorm)
    return optimizer


# ================================

def main(argv):
    del argv

    if FLAGS.CUDA_VISIBLE_DEVICES is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.CUDA_VISIBLE_DEVICES
        logging.info(f'\x1b[1;32;m\t gpu:{FLAGS.CUDA_VISIBLE_DEVICES} specified!\x1b[0m')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    if not any([FLAGS.do_train, FLAGS.do_test, FLAGS.do_predict]):
        raise ValueError(f"At least one mode is set to be true.")

    print_configuration_op(FLAGS)
    reset_seed(FLAGS.random_seed)

    output_dir = f'output_dir/{FLAGS.dataset}/{FLAGS.model_name}'

    # ==============================
    # load train / eval data
    # ==============================
    train_data, val_data, vocab_dict, idx_word, ref_val = create_dataset(dataset=FLAGS.dataset,
                                                                         raw_data_path=FLAGS.raw_data_path,
                                                                         cnn_type=FLAGS.cnn_type,
                                                                         reset_cache=FLAGS.reset_cache,
                                                                         max_seq_len=FLAGS.max_seq_len + 1,
                                                                         train_bsz=FLAGS.train_bsz,
                                                                         val_bsz=FLAGS.val_bsz,
                                                                         debug=FLAGS.debug)

    if FLAGS.max_num_words is not None:
        vocab_size = max(FLAGS.max_num_words, len(vocab_dict))
    else:
        vocab_size = len(vocab_dict)
    logging.info(f"vocabulary size: {vocab_size}")

    # output dir
    # ==============================
    hyperparams_to_tune = {
        'cap_len': FLAGS.max_seq_len,
        'lr': FLAGS.lr,
        # 'drpt': FLAGS.dropout_rate,
    }
    model_repr = '_'.join([f"{k}{v}" for k, v in hyperparams_to_tune.items()])
    # save_dir = os.path.join(output_dir, model_repr)
    save_dir = os.path.join(output_dir, model_repr, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(save_dir, exist_ok=True)

    # load model_config
    # ===============================
    model_config = create_model_config(FLAGS.model_name, vocab_size, vocab_dict, idx_word, FLAGS.num_epochs, model_repr)

    # optimizer
    # ==============================
    optimizer = create_optimizer(optim_name=FLAGS.optim_name, lr=FLAGS.lr,
                                 clipnorm=FLAGS.clipnorm if FLAGS.do_clip else None)

    # model
    # ==============================
    model = create_model(FLAGS.model_name, model_config, optimizer)
    ckpt_dir = os.path.join(save_dir, f'best-model-{model_repr}')
    # train
    if FLAGS.do_train:
        model.train_loop(train_data, val_data, FLAGS, ckpt_dir, ref_val)

    if FLAGS.do_test:
        test_data, vocab_dict, idx_word, test_ref = load_test_data(dataset=FLAGS.dataset,
                                                                   raw_data_path=FLAGS.raw_data_path,
                                                                   cnn_type=FLAGS.cnn_type,
                                                                   reset_cache=FLAGS.reset_cache,
                                                                   max_seq_len=FLAGS.max_seq_len + 1,
                                                                   test_bsz=FLAGS.test_bsz,
                                                                   debug=FLAGS.debug)
        model.inference(test_data, FLAGS, ckpt_dir, test_ref)

        # plt.plot(loss_plot)
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.title('Loss Plot')
        # plt.show()


if __name__ == '__main__':
    img_cap_flags()
    app.run(main)
