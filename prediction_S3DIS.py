from os import makedirs
from os.path import exists, join
from helper_ply import write_ply
from helper_tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import os

def log_out(out_str, log_f_out):
    log_f_out.write(out_str + '\n')
    log_f_out.flush()
    print(out_str)

class ModelTester:
    def __init__(self, model, dataset, threshold, span, restore_snap=None):
        self.threshold = threshold
        self.span = span
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        self.test_log_path = os.path.join(dataset.experiment_dir, 'prediction')
        self.test_save_path = os.path.join(self.test_log_path, 'val_preds')
        makedirs(self.test_save_path) if not exists(self.test_save_path) else None

        log_file_path = os.path.join(dataset.experiment_dir, 'prediction', 'log' + '.txt')
        self.Log_file = open(log_file_path, 'a')
        print('test log dir:', log_file_path)

        # Create a session for running Ops on the Graph.
        on_cpu = False
        if on_cpu:
            c_proto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            c_proto = tf.ConfigProto()
            c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.sess.run(tf.global_variables_initializer())
        if restore_snap is not None:
            self.saver.restore(self.sess, restore_snap)
            log_out("Model restored from " + restore_snap, self.Log_file)

        self.prob_logits = tf.nn.softmax(model.logits)

        # Initiate global prediction over all test clouds
        self.test_probs = [np.zeros(shape=[l.shape[0], model.config.num_classes], dtype=np.float32)
                           for l in dataset.input_labels['validation']]
        self.probs_var = [np.zeros(shape=[l.shape[0], model.config.num_classes], dtype=np.float32)
                           for l in dataset.input_labels['validation']]
        self.probs_store = [[[] for i in range(l.shape[0])] for l in dataset.input_labels['validation']]

    def test(self, model, dataset, num_votes=10):

        # Smoothing parameter for votes
        test_smooth = 0.95

        # Initialise iterator with validation/test data
        self.sess.run(dataset.val_init_op)

        # Number of points per class in validation set
        val_proportions = np.zeros(model.config.num_classes, dtype=np.float32)
        i = 0
        for label_val in dataset.label_values:
            if label_val not in dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_val) for labels in dataset.val_labels])
                i += 1
        test_path = self.test_log_path
        makedirs(test_path) if not exists(test_path) else None
        makedirs(join(test_path, 'pseudo_label')) if not exists(join(test_path, 'pseudo_label')) else None
        step_id = 0
        epoch_id = 0
        last_min = -0.5

        while last_min < num_votes:
            try:
                ops = (self.prob_logits,
                       model.labels,
                       model.inputs['input_inds'],
                       model.inputs['cloud_inds'],
                       )

                stacked_probs, stacked_labels, point_idx, cloud_idx = self.sess.run(ops, {model.is_training: False})
                correct = np.sum(np.argmax(stacked_probs, axis=1) == stacked_labels)
                acc = correct / float(np.prod(np.shape(stacked_labels)))
                log_out('step' + str(step_id) + ' acc:' + str(acc), self.Log_file)
                stacked_probs = np.reshape(stacked_probs, [model.config.val_batch_size, model.config.num_points,
                                                           model.config.num_classes])

                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]
                    p_idx = point_idx[j, :]
                    c_i = cloud_idx[j][0]
                    self.test_probs[c_i][p_idx] = test_smooth * self.test_probs[c_i][p_idx] + (1 - test_smooth) * probs
                    for i in range(40960):
                        self.probs_store[c_i][p_idx[i]] += [probs[i]]
                step_id += 1

            except tf.errors.OutOfRangeError:
                new_min = np.min(dataset.min_possibility['validation'])
                log_out('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min), self.Log_file)

                if True:
                    last_min += 1
                    confusion_list = []
                    num_val = len(dataset.input_labels['validation'])

                    sum_point = 0
                    pseudo_point = 0
                    sum_correct = 0
                    sum_all = 0
                    for i_test in range(num_val):
                        probs = self.test_probs[i_test]
                        probs2 = self.np_normalized(probs)
                        prob_store = self.probs_store[i_test]
                        for i in range(np.array(prob_store).shape[0]):
                            each_var = np.var(prob_store[i], axis=0)
                            self.probs_var[i_test][i] = self.probs_var[i_test][i] - each_var
                        probs2 = np.multiply(probs2, np.exp(self.probs_var[i_test]))
                        probs2 = self.np_normalized(probs2)

                        p_t = np.max(probs2, axis=0) - self.span
                        p_t[np.argwhere(p_t < self.threshold - self.span)] = self.threshold - self.span
                        pt_c = np.repeat(np.reshape(p_t, [1, -1]), repeats=probs2.shape[0], axis=0)
                        probs2 = probs2 - pt_c
                        uncertainty_idx = np.argwhere(np.max(probs2, axis=1) < 0)
                        uncertainty_idx = np.reshape(uncertainty_idx, [-1])
                        sum_point += probs2.shape[0]
                        pseudo_point += uncertainty_idx.shape[0]

                        preds = dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)
                        for id in uncertainty_idx :
                            preds[id] = 13

                        labels = dataset.input_labels['validation'][i_test]
                        labels_p  = np.delete(labels, uncertainty_idx)
                        preds_p = np.delete(preds, uncertainty_idx)
                        correct_preds = np.sum(labels_p == preds_p)
                        sum_preds = np.prod(np.shape(preds_p))
                        acc_preds = correct_preds / float(sum_preds)
                        log_out('proportion = {:f}, acc_pred = {:.3f}'.format(1 - uncertainty_idx.shape[0] / probs2.shape[0], acc_preds), self.Log_file)
                        sum_correct += correct_preds
                        sum_all += sum_preds

                        cloud_name = dataset.input_names['validation'][i_test]
                        ascii_name = join(test_path, 'pseudo_label', cloud_name)
                        np.save(ascii_name, preds)
                        log_out(ascii_name + ' has saved', self.Log_file)

                    pseudo_point = sum_point - pseudo_point
                    log_out('sum: {:d}, pseudo labels: {:d}, proportion = {:f}'.format(sum_point, pseudo_point, pseudo_point / sum_point), self.Log_file)
                    log_out('sum correct: {:d}, all: {:d}, correct acc:  = {:.3f}'.format(sum_correct, sum_all, sum_correct / sum_all), self.Log_file)
                    return

        self.Log_file.close()
        return

    def np_normalized(self, z):
        a = z / np.reshape(np.sum(z, axis=1), [-1,1])
        return a
