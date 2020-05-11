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
@file: model.py
@time: 5/6/20 9:40 PM
@desc：       
               
'''
from utils.utils import *
from layers.layers import *


class TrainTemplate(object):
    """ base model template """

    def __init__(self, model_config, optimizer):
        self.model_config = model_config
        self.optimizer = optimizer
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.loss = 0.
        self.accuracy = 0.
        self.dropout_rate = 0.
        self.training = None
        self.num_epochs = model_config.num_epochs
        self.vocab_dict = model_config.vocab_dict
        self.idx_word = model_config.idx_word
        self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
        self.scorer = measure_score

    def loss_function(self, label, pred):
        mask = tf.math.logical_not(tf.math.equal(label, 0))
        loss_ = self.loss_object(label, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    def train(self):
        self.training = True

    def val(self):
        self.training = False

    def train_step(self, x, y):
        loss = 0.
        hidden = self.decoder.reset_state(batch_size=y.shape[0])
        with tf.GradientTape() as tape:
            features = self.encoder(x)
            dec_input = tf.expand_dims([self.vocab_dict[START]] * y.shape[0], 1)
            for t in range(1, y.shape[1]):
                pred, hidden, _ = self.decoder(dec_input, features, hidden)
                loss += self.loss_function(y[:, t], pred)
                dec_input = tf.expand_dims(y[:, t], 1)
        batch_loss = loss / int(y.shape[1])

        trainable_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # tf.clip_by_global_norm(gradients, 5.)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.train_loss_metric(batch_loss)
        return loss, batch_loss

    def test_step(self, x, y):
        loss = 0.
        hypos = []
        hidden = self.decoder.reset_state(batch_size=y.shape[0])
        with tf.GradientTape():
            features = self.encoder(x)
            dec_input = tf.expand_dims([self.vocab_dict[START]] * y.shape[0], 1)
            for t in range(1, y.shape[1]):
                pred, hidden, _ = self.decoder(dec_input, features, hidden)
                loss += self.loss_function(y[:, t], pred)
                pred_ids = tf.argmax(pred, axis=1)  # greedy
                hypos.append(pred_ids.numpy())
                dec_input = tf.expand_dims(pred_ids, 1)
        batch_loss = loss / int(y.shape[1])
        self.test_loss_metric(batch_loss)
        hypos = np.vstack(hypos).transpose()
        return hypos, batch_loss

    def train_loop(self, train_data, val_data, FLAGS, ckpt_dir, val_refs):
        if FLAGS.enable_function:
            self.train_step = tf.function(self.train_step)
            # self.test_step = tf.function(self.test_step)

        # model save
        # ==============================
        start_epoch = 0
        if FLAGS.save_models:
            ckpt = tf.train.Checkpoint(encoder=self.encoder,
                                       decoder=self.decoder,
                                       optimizer=self.optimizer)
            ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)
            if ckpt_manager.latest_checkpoint:
                start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
                ckpt.restore(ckpt_manager.latest_checkpoint).assert_consumed()

        # early stopping
        best_scores = {k: 0. for k in ('BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE', 'CIDEr')}
        best_val_loss = np.inf
        early_stop = best_step = 0  # early stop patience

        cur_step = 0
        for epoch in range(start_epoch, self.num_epochs):
            self.train_loss_metric.reset_states()
            start = time.time()
            train_loss = 0.
            for (batch, (imgid, img, cap)) in enumerate(train_data):
                loss, batch_loss = self.train_step(img, cap)
                train_loss += batch_loss

                cur_step += 1
                if (batch + 1) % FLAGS.print_every == 0:
                    print(f'Epoch {epoch + 1:04} Batch {batch + 1:8} Loss {loss.numpy() / int(cap.shape[1]):8.4f}')

                # evaluate
                if FLAGS.debug or cur_step % FLAGS.val_every == 0:
                    print(f"{'*' * 50}  Evaluating {cur_step}-th step {'*' * 50}")
                    val_loss = 0.
                    self.test_loss_metric.reset_states()
                    val_hypos = {}
                    for (val_batch, (val_imgid, val_img, val_ref)) in enumerate(val_data):
                        val_hypo, batch_loss = self.test_step(val_img, val_ref)
                        val_loss += batch_loss
                        val_hypos.update({k: v for k, v in zip(val_imgid.numpy().tolist(), val_hypo.tolist())})
                    avg_val_loss = val_loss / (val_batch + 1)
                    m_val_loss = self.test_loss_metric.result().numpy()
                    eval_scores = evaluate(val_hypos, val_refs, self.idx_word, scorer=self.scorer)
                    print(f"\x1b[1;36;m Step {cur_step},  "
                          f"avg_val_loss: {avg_val_loss:.4f}, "
                          f"m_val_loss: {m_val_loss:.4f} "
                          f"\n {eval_scores} \x1b[0m")

                    # early stopping
                    if m_val_loss < best_val_loss or any([eval_scores[k] > best_scores[k] for k in eval_scores]):
                        best_val_loss = m_val_loss
                        best_step = cur_step
                        for k in eval_scores:
                            if eval_scores[k] > best_scores[k]:
                                best_scores[k] = eval_scores[k]
                                print(f"\x1b[1;32;m Step {cur_step}, "
                                      f"best {k}: {best_scores[k]:8.4f} \x1b[0m")
                        if FLAGS.save_models:
                            ckpt_manager.save()
                            if FLAGS.debug:
                                if cur_step > 1:
                                    return
                    else:
                        early_stop += 1
                        if early_stop >= FLAGS.tol:
                            logging.info(
                                f"\x1b[1;34;m \t Early stopping at step {best_step}, "
                                f"best_val_loss= {best_val_loss}, "
                                f"best_scores={best_scores} \x1b[0m")
                            # save log
                            save_log(os.path.dirname(ckpt_dir) + '/test_results.txt',
                                     {'model': self.model_config.model_repr,
                                      'best_step': best_step,
                                      'best_val_loss': best_val_loss,
                                      'best_scores': best_scores})
                            return

            print(f'Epoch {epoch + 1} Loss {self.train_loss_metric.result().numpy():.6f}')
            print(f'Time cost for 1 epoch {time.time() - start:.2f} sec\n')

    def inference(self, test_data, FLAGS, ckpt_dir, test_refs):
        print("Finish training! Starting to test ...")
        # restore model
        ckpt = tf.train.Checkpoint(encoder=self.encoder,
                                   decoder=self.decoder,
                                   optimizer=self.optimizer)
        ckpt_dir = ckpt_dir if FLAGS.restore_dir is None else FLAGS.restore_dir
        ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=5)
        ckpt.restore(ckpt_manager.latest_checkpoint)
        # test
        test_loss = 0.
        self.test_loss_metric.reset_states()
        test_hypos = {}
        for (test_batch, (test_imgid, test_img, test_ref)) in enumerate(test_data):
            test_hypo, batch_loss = self.test_step(test_img, test_ref)
            test_loss += batch_loss
            test_hypos.update({k: v for k, v in zip(test_imgid.numpy().tolist(), test_hypo.tolist())})
        avg_test_loss = test_loss / (test_batch + 1)
        m_test_loss = self.test_loss_metric.result().numpy()
        gen_captions_path = os.path.join(os.path.dirname(ckpt_dir), 'gen_captions.txt')
        test_scores = evaluate(test_hypos, test_refs, self.idx_word, scorer=self.scorer, save_path=gen_captions_path)
        print(f"\x1b[1;31;m"
              f"avg_test_loss: {avg_test_loss:8.4f}, "
              f"m_test_loss: {m_test_loss:8.4f} "
              f"\n test scores:\n"
              f" {test_scores} \x1b[0m")
        test_scores.update({'test_loss': m_test_loss})
        test_log_path = os.path.join(os.path.dirname(ckpt_dir), 'test.log')
        try:
            save_log(test_log_path, test_scores)
        except IOError as e:
            print(e)
        finally:
            print(f'{test_log_path} saved!')

    def __repr__(self):
        return 'Base model cls @CYK'


class DistributeTrain(TrainTemplate):
    """ distributed base model"""

    def __init__(self, model_config, optimizer, strategy):
        super().__init__(model_config, optimizer)
        self.strategy = strategy

    def train_step(self, x, y):
        loss = 0.
        hidden = self.decoder.reset_state(batch_size=y.shape[0])
        with tf.GradientTape() as tape:
            features = self.encoder(x)
            dec_input = tf.expand_dims([self.vocab_dict[START]] * y.shape[0], 1)
            for t in range(1, y.shape[1]):
                pred, hidden, _ = self.decoder(dec_input, features, hidden)
                loss += self.loss_function(y[:, t], pred)
                dec_input = tf.expand_dims(y[:, t], 1)
        batch_loss = loss / int(y.shape[1])

        trainable_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # tf.clip_by_global_norm(gradients, 5.)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.train_loss_metric(batch_loss)
        return loss, batch_loss

    def test_step(self, x, y):
        loss = 0.
        hypos = []
        hidden = self.decoder.reset_state(batch_size=y.shape[0])
        with tf.GradientTape():
            features = self.encoder(x)
            dec_input = tf.expand_dims([self.vocab_dict[START]] * y.shape[0], 1)
            for t in range(1, y.shape[1]):
                pred, hidden, _ = self.decoder(dec_input, features, hidden)
                loss += self.loss_function(y[:, t], pred)
                pred_ids = tf.argmax(pred, axis=1)  # greedy
                hypos.append(pred_ids.numpy())
                dec_input = tf.expand_dims(pred_ids, 1)
        batch_loss = loss / int(y.shape[1])
        self.test_loss_metric(batch_loss)
        hypos = np.vstack(hypos).transpose()
        return hypos, batch_loss

    def distributed_train_step(self, x, y):
        losses, per_replica_losses = self.strategy.experimental_run_v2(self.train_step, args=(x, y,))
        return losses, self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

    def distributed_test_step(self, x, y):
        return self.strategy.experimental_run_v2(self.test_step, args=(x, y,))

    def train_loop(self, train_data, val_data, FLAGS, ckpt_dir, val_refs):
        if FLAGS.enable_function:
            self.train_step = tf.function(self.train_step)
            # self.test_step = tf.function(self.test_step)

        # model save
        # ==============================
        start_epoch = 0
        if FLAGS.save_models:
            # ckpt_prefix = os.path.join(ckpt_dir, 'ckpt')
            ckpt = tf.train.Checkpoint(encoder=self.encoder,
                                       decoder=self.decoder,
                                       optimizer=self.optimizer)
            ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=5)
            # if ckpt_manager.latest_checkpoint:
            #     start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
            #     status = ckpt.restore(ckpt_manager.latest_checkpoint)
            #     status.assert_existing_objects_matched()

        # early stopping
        best_scores = {k: 0. for k in ('BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE', 'CIDEr')}
        best_val_loss = np.inf
        early_stop = best_step = 0  # early stop patience

        cur_step = 0
        for epoch in range(start_epoch, self.num_epochs):
            self.train_loss_metric.reset_states()
            start = time.time()
            train_loss = 0.
            for (batch, (imgid, img, cap)) in enumerate(train_data):
                # loss, batch_loss = self.train_step(img, cap)
                loss, batch_loss = self.distributed_train_step(img, cap)
                train_loss += batch_loss

                cur_step += 1
                if FLAGS.debug or (batch + 1) % FLAGS.print_every == 0:
                    if self.strategy.num_replicas_in_sync > 1:
                        loss = sum(per_replica_loss.numpy() for per_replica_loss in loss.values)
                        loss /= (cap.values[0].shape[1] * self.strategy.num_replicas_in_sync)
                    else:
                        loss = loss.numpy()
                        loss /= cap.shape[1]
                    print(f'Epoch {epoch + 1:04} | Batch {batch + 1:8} | '
                          f'Loss {loss:8.4f}')

                # evaluate
                if FLAGS.debug or cur_step % FLAGS.val_every == 0:
                    print(f"{'*' * 50}  Evaluating {cur_step}-th step {'*' * 50}")
                    val_loss = 0.
                    self.test_loss_metric.reset_states()
                    val_hypos = {}
                    for (val_batch, (val_imgid, val_img, val_ref)) in enumerate(val_data):
                        val_hypo, batch_loss = self.distributed_test_step(val_img, val_ref)
                        if self.strategy.num_replicas_in_sync > 1:
                            val_loss += sum(per_replica_loss.numpy() for per_replica_loss in batch_loss.values)
                            for i in range(self.strategy.num_replicas_in_sync):
                                imgid = val_imgid.values[i]
                                hypo = val_hypo.values[i]
                                val_hypos.update({k: v for k, v in zip(imgid.numpy().tolist(), hypo.tolist())})
                            # val_hypos.extend(list(val_hypo.values))
                            # val_refs.extend([ref[:, 1:] for ref in val_ref.values])
                        else:
                            val_loss += batch_loss
                            val_hypos.update({k: v for k, v in zip(val_imgid.numpy().tolist(), val_hypo.tolist())})
                            # val_hypos.append(val_hypo)
                            # val_refs.append(val_ref[:, 1:])
                    avg_val_loss = val_loss / ((val_batch + 1) * self.strategy.num_replicas_in_sync)
                    m_val_loss = self.test_loss_metric.result().numpy()
                    eval_scores = evaluate(val_hypos, val_refs, self.idx_word, scorer=self.scorer)
                    print(f"\x1b[1;36;m Step {cur_step},  "
                          f"m_val_loss: {m_val_loss:.4f} "
                          f"avg_val_loss: {avg_val_loss} "
                          f"\n {eval_scores} \x1b[0m")

                    # early stopping
                    if m_val_loss <= best_val_loss or any([eval_scores[k] > best_scores[k] for k in eval_scores]):
                        best_val_loss = m_val_loss
                        best_step = cur_step
                        for k in eval_scores:
                            if eval_scores[k] > best_scores[k]:
                                best_scores[k] = eval_scores[k]
                                print(f"\x1b[1;32;m Step {cur_step}, "
                                      f"best {k}: {best_scores[k]:8.4f} \x1b[0m")
                        if FLAGS.save_models:
                            ckpt_manager.save()
                            if FLAGS.debug:
                                if cur_step > 1:
                                    return
                    else:
                        early_stop += 1
                        if early_stop >= FLAGS.tol:
                            logging.info(
                                f"\x1b[1;34;m \t Early stopping at step {best_step}, "
                                f"best_val_loss= {best_val_loss:8.4f}, "
                                f"best_scores={best_scores} \x1b[0m")
                            # save log
                            save_log(os.path.dirname(ckpt_dir) + '/test_results.txt',
                                     {'model': self.model_config.model_repr,
                                      'best_step': best_step,
                                      'best_val_loss': best_val_loss,
                                      'best_scores': best_scores})
                            return

            print(f'Epoch {epoch + 1} Loss {self.train_loss_metric.result().numpy():.6f}')
            print(f'Time cost for 1 epoch {time.time() - start:.2f} sec\n')

    def inference(self, test_data, FLAGS, ckpt_dir, test_refs):
        print("Finish training! Starting to test ...")
        # restore model
        ckpt = tf.train.Checkpoint(encoder=self.encoder,
                                   decoder=self.decoder,
                                   optimizer=self.optimizer)
        ckpt_dir = ckpt_dir if FLAGS.restore_dir is None else FLAGS.restore_dir
        ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir,
                                                  max_to_keep=3)
        status = ckpt.restore(ckpt_manager.latest_checkpoint)
        status.assert_consumed()

        # test
        test_loss = 0.
        self.test_loss_metric.reset_states()
        test_hypos = {}
        for (test_batch, (test_imgid, test_img, test_ref)) in enumerate(test_data):
            test_hypo, batch_loss = self.distributed_test_step(test_img, test_ref)
            if self.strategy.num_replicas_in_sync > 1:
                test_loss += sum(per_replica_loss.numpy() for per_replica_loss in batch_loss.values)
                for i in range(self.strategy.num_replicas_in_sync):
                    imgid = test_imgid.values[i]
                    hypo = test_hypo.values[i]
                    test_hypos.update({k: v for k, v in zip(imgid.numpy().tolist(), hypo.tolist())})
            else:
                test_loss += batch_loss
                test_hypos.update({k: v for k, v in zip(test_imgid.numpy().tolist(), test_hypo.tolist())})

        avg_test_loss = test_loss / ((test_batch + 1) * self.strategy.num_replicas_in_sync)
        m_test_loss = self.test_loss_metric.result().numpy()
        gen_captions_path = os.path.join(os.path.dirname(ckpt_dir), 'gen_captions.txt')
        test_scores = evaluate(test_hypos, test_refs, self.idx_word, scorer=self.scorer, save_path=gen_captions_path)
        print(f"\x1b[1;31;m"
              f"avg_test_loss: {avg_test_loss:8.4f}, "
              f"m_test_loss: {m_test_loss:8.4f} "
              f"\n test scores:\n"
              f" {test_scores} \x1b[0m")
        test_scores.update({'test_loss': m_test_loss})
        test_log_path = os.path.join(os.path.dirname(ckpt_dir), 'test.log')
        try:
            save_log(test_log_path, test_scores)
        except IOError as e:
            print(e)
        finally:
            print(f'{test_log_path} saved!')

    def __repr__(self):
        return "NIC model"
