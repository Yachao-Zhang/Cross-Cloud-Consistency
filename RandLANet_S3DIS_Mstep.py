import os
from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import helper_tf_util
import time


def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)

class Network:
    def __init__(self, dataset, config):
        flat_inputs = dataset.flat_inputs
        self.config = config
        # Path of the result folder
        if self.config.saving:
            if self.config.saving_path is None:
                self.saving_path = dataset.checkpoints_dir
            else:
                self.saving_path = self.config.saving_path
            makedirs(self.saving_path) if not exists(self.saving_path) else None

        with tf.variable_scope('inputs'):
            self.inputs = dict()
            num_layers = self.config.num_layers
            self.inputs['xyz'] = flat_inputs[:num_layers]
            self.inputs['neigh_idx'] = flat_inputs[num_layers: 2 * num_layers]
            self.inputs['sub_idx'] = flat_inputs[2 * num_layers:3 * num_layers]
            self.inputs['interp_idx'] = flat_inputs[3 * num_layers:4 * num_layers]
            self.inputs['features'] = flat_inputs[4 * num_layers]
            self.inputs['labels'] = flat_inputs[4 * num_layers + 1]
            self.inputs['labels_gt'] = flat_inputs[4 * num_layers + 2]
            self.inputs['input_inds'] = flat_inputs[4 * num_layers + 3]
            self.inputs['cloud_inds'] = flat_inputs[4 * num_layers + 4]

            self.labels = self.inputs['labels']
            self.labels_gt = self.inputs['labels_gt']
            self.is_training = tf.placeholder(tf.bool, shape=())
            self.training_step = 1
            self.training_epoch = 0
            self.correct_prediction = 0
            self.accuracy = 0
            self.mIou_list = [0]
            self.class_weights = DP.get_class_weights(dataset.name)
            if self.config.saving:
                self.Log_file = open(dataset.experiment_dir + '/log_train.txt', 'a')
                self.Log_file.write(' '.join(["config.%s = %s\n" % (k, v) for k, v in self.config.__dict__.items() if not k.startswith('__')]))

        with tf.variable_scope('layers'):
            self.logits, self.embedding = self.inference(self.inputs, self.is_training)

        #####################################################################
        # Ignore the invalid point (unlabeled) when calculating the loss #
        #####################################################################
        with tf.variable_scope('loss'):
            d = self.embedding.get_shape()[-1].value
            valid_labels_ps, valid_logits_ps, _ = self.data_prep(self.logits, self.embedding, self.labels, d)
            valid_labels_gt, valid_logits_gt, _ = self.data_prep(self.logits, self.embedding, self.labels_gt, d)
            label_loss_ps = self.get_loss(valid_logits_ps, valid_labels_ps)
            label_loss_gt = self.get_loss(valid_logits_gt, valid_labels_gt)
            tf.summary.scalar('label_loss_ps', label_loss_ps)
            tf.summary.scalar('label_loss_gt', label_loss_gt)

            #############-pull-push-#############
            contrastive_loss = 0
            # Cross-scene Contrastive
            for i in range(self.config.batch_size // 2):
                per_logits1 = self.logits[i:i + 1, :, :]  # [1,N,C]
                per_logits2 = self.logits[self.config.batch_size // 2 + i:self.config.batch_size // 2 + i + 1, :, :]
                per_embedding1 = self.embedding[i:i + 1, :, :]  # [1,N,d]
                per_embedding2 = self.embedding[self.config.batch_size // 2 + i:self.config.batch_size // 2 + i + 1, :, :]
                per_labels1 = self.labels[i:i + 1, :]
                per_labels2 = self.labels[self.config.batch_size // 2 + i:self.config.batch_size // 2 + i + 1, :]
                valid_labels1, _, valid_embedding1 = self.data_prep(per_logits1, per_embedding1, per_labels1, d)
                valid_labels2, _, valid_embedding2 = self.data_prep(per_logits2, per_embedding2, per_labels2, d)
                single_pull_ploss, single_push_dloss = self.pull_and_push_loss(valid_embedding1, valid_embedding2,
                                                                              valid_labels1, valid_labels2, d)
                contrastive_loss = contrastive_loss + 0.5 * single_pull_ploss + 0.001 * single_push_dloss
                tf.summary.scalar('single_pull_ploss', single_pull_ploss)
                tf.summary.scalar('single_push_dloss', single_push_dloss)

            self.loss = label_loss_ps + label_loss_gt + contrastive_loss
            self.logits = tf.reshape(self.logits, [-1, self.config.num_classes])
            self.labels = tf.reshape(self.labels, [-1])

        with tf.variable_scope('optimizer'):
            self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('results'):
            self.correct_prediction = tf.nn.in_top_k(valid_logits_ps, valid_labels_ps, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.prob_logits = tf.nn.softmax(self.logits)

            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)

        if True:
            my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            my_vars = [i for i in my_vars if 'layers/Encoder_layer_' in i.name]
            self.saver = tf.train.Saver(my_vars, max_to_keep=100)
            c_proto = tf.ConfigProto()
            c_proto.gpu_options.allow_growth = True
            self.sess = tf.Session(config=c_proto)
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(dataset.tensorboard_log_dir, self.sess.graph)
            self.sess.run(tf.global_variables_initializer())
            print('Every times at running?')
            # Load trained model
            checkpoints_dir = self.config.load_dir
            snap_path = os.path.join(checkpoints_dir, 'snapshots')
            snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
            chosen_step = np.sort(snap_steps)[-1]
            chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
            restore_snap = chosen_snap
            if restore_snap is not None:
                self.saver.restore(self.sess, restore_snap)
                print("Model restored from " + restore_snap)
            my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            self.saver = tf.train.Saver(my_vars, max_to_keep=100)

    def data_prep(self, logits, embedding, labels, d):
        logits = tf.reshape(logits, [-1, self.config.num_classes]) #[B*N,C]
        embedding = tf.reshape(embedding, [-1, d]) #[B*N,d]
        labels = tf.reshape(labels, [-1]) #[B*N]

        # Boolean mask of points that should be ignored
        ignored_bool = tf.zeros_like(labels, dtype=tf.bool)
        for ign_label in self.config.ignored_label_inds:
            ignored_bool = tf.logical_or(ignored_bool, tf.equal(labels, ign_label))

        # Collect logits and labels that are not ignored
        valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
        valid_logits = tf.gather(logits, valid_idx, axis=0)
        valid_embedding = tf.gather(embedding, valid_idx, axis=0)
        valid_labels_init = tf.gather(labels, valid_idx, axis=0)

        # Reduce label values in the range of logit shape
        reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
        inserted_value = tf.zeros((1,), dtype=tf.int32) #[0]
        for ign_label in self.config.ignored_label_inds:
            reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
        valid_labels = tf.gather(reducing_list, valid_labels_init)
        return valid_labels, valid_logits, valid_embedding

    def inference(self, inputs, is_training):
        d_out = self.config.d_out
        feature = inputs['features']
        feature = tf.layers.dense(feature, 8, activation=None, name='fc0')
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        feature = tf.expand_dims(feature, axis=2)

        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_block(feature, inputs['xyz'][i], inputs['neigh_idx'][i], d_out[i],
                                                 'Encoder_layer_' + str(i), is_training)
            f_sampled_i = self.random_sample(f_encoder_i, inputs['sub_idx'][i])
            feature = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        feature = helper_tf_util.conv2d(f_encoder_list[-1], f_encoder_list[-1].get_shape()[3].value, [1, 1],
                                        'decoder_0',
                                        [1, 1], 'VALID', True, is_training)

        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(feature, inputs['interp_idx'][-j - 1])
            f_decoder_i = helper_tf_util.conv2d_transpose(tf.concat([f_encoder_list[-j - 2], f_interp_i], axis=3),
                                                          f_encoder_list[-j - 2].get_shape()[-1].value, [1, 1],
                                                          'Decoder_layer_' + str(j), [1, 1], 'VALID', bn=True,
                                                          is_training=is_training)
            feature = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################

        f_layer_fc1 = helper_tf_util.conv2d(f_decoder_list[-1], 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)
        f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training)
        f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc', [1, 1], 'VALID', False,
                                            is_training, activation_fn=None)
        f_layer_embed = helper_tf_util.conv2d(f_layer_drop, 32, [1, 1], 'fce', [1, 1], 'VALID', True, is_training)
        f_out = tf.squeeze(f_layer_fc3, [2])
        return f_out, tf.squeeze(f_layer_embed, [2])

    def train(self, dataset):
        log_out('test Area:{}'.format(dataset.val_split), self.Log_file)
        log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)
        self.best_epoch = 0
        self.sess.run(dataset.train_init_op)
        while self.training_epoch < self.config.iter_epoch:
            t_start = time.time()
            try:
                ops = [self.train_op,
                       self.extra_update_ops,
                       self.merged,
                       self.loss,
                       self.logits,
                       self.labels,
                       self.accuracy]
                _, _, summary, l_out, probs, labels, acc = self.sess.run(ops, {self.is_training: True})
                self.train_writer.add_summary(summary, self.training_step)
                t_end = time.time()
                if self.training_step % 50 == 0:
                    message = 'Step {:08d} L_out={:5.3f} Acc={:4.2f} ''---{:8.2f} ms/batch'
                    log_out(message.format(self.training_step, l_out, acc, 1000 * (t_end - t_start)), self.Log_file)
                self.training_step += 1

            except tf.errors.OutOfRangeError:
                m_iou = self.evaluate(dataset)
                if m_iou > np.max(self.mIou_list):
                    # Save the best model
                    self.best_epoch = self.training_epoch
                    snapshot_directory = join(self.saving_path, 'snapshots')
                    makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                    self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step)
                self.mIou_list.append(m_iou)
                log_out('Best m_IoU is: {:5.3f}, epoch: {}'.format(max(self.mIou_list), self.best_epoch), self.Log_file)

                self.training_epoch += 1
                self.sess.run(dataset.train_init_op)

                # Update learning rate
                op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                           self.config.lr_decays[self.training_epoch]))
                self.sess.run(op)
                log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)

            except tf.errors.InvalidArgumentError as e:
                print('Caught a NaN error :')
                print(e.error_code)
                print(e.message)
                print(e.op)
                print(e.op.name)
                print([t.name for t in e.op.inputs])
                print([t.name for t in e.op.outputs])
                a = 1 / 0

        print('finished')
        self.sess.close()

    def evaluate(self, dataset):
        # Initialise iterator with validation data
        self.sess.run(dataset.val_init_op)

        gt_classes = [0 for _ in range(self.config.num_classes)]
        positive_classes = [0 for _ in range(self.config.num_classes)]
        true_positive_classes = [0 for _ in range(self.config.num_classes)]
        val_total_correct = 0
        val_total_seen = 0

        for step_id in range(self.config.val_steps):
            if step_id % 50 == 0:
                print(str(step_id) + ' / ' + str(self.config.val_steps))
            try:
                ops = (self.prob_logits, self.labels, self.accuracy)
                stacked_prob, labels, acc = self.sess.run(ops, {self.is_training: False})
                pred = np.argmax(stacked_prob, 1)
                if not self.config.ignored_label_inds:
                    pred_valid = pred
                    labels_valid = labels
                else:
                    invalid_idx = np.where(labels == self.config.ignored_label_inds)[0]
                    labels_valid = np.delete(labels, invalid_idx)
                    pred_valid = np.delete(pred, invalid_idx)

                correct = np.sum(pred_valid == labels_valid)
                val_total_correct += correct
                val_total_seen += len(labels_valid)

                conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.config.num_classes, 1))
                gt_classes += np.sum(conf_matrix, axis=1)
                positive_classes += np.sum(conf_matrix, axis=0)
                true_positive_classes += np.diagonal(conf_matrix)

            except tf.errors.OutOfRangeError:
                break

        iou_list = []
        for n in range(0, self.config.num_classes, 1):
            iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
            iou_list.append(iou)
        mean_iou = sum(iou_list) / float(self.config.num_classes)

        log_out('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)), self.Log_file)
        log_out('mean IOU:{}'.format(mean_iou), self.Log_file)

        mean_iou = 100 * mean_iou
        log_out('Mean IoU = {:.1f}%'.format(mean_iou), self.Log_file)
        s = '{:5.2f} | '.format(mean_iou)
        for IoU in iou_list:
            s += '{:5.2f} '.format(100 * IoU)
        log_out('-' * len(s), self.Log_file)
        log_out(s, self.Log_file)
        log_out('-' * len(s) + '\n', self.Log_file)
        return mean_iou

    def get_loss(self, logits, labels):
        # calculate the weighted cross entropy according to the inverse frequency
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
        output_loss = tf.reduce_mean(unweighted_losses)
        return output_loss

    def dilated_res_block(self, feature, xyz, neigh_idx, d_out, name, is_training):
        f_pc = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_pc = self.building_block(xyz, f_pc, neigh_idx, d_out, name + 'LFA', is_training)
        f_pc = helper_tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training,
                                     activation_fn=None)
        shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
                                         activation_fn=None, bn=True, is_training=is_training)
        return tf.nn.leaky_relu(f_pc + shortcut)

    def building_block(self, xyz, feature, neigh_idx, d_out, name, is_training):
        d_in = feature.get_shape()[-1].value
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)
        f_xyz = helper_tf_util.conv2d(f_xyz, d_in, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out // 2, name + 'att_pooling_1', is_training)

        f_xyz = helper_tf_util.conv2d(f_xyz, d_out // 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(f_pc_agg, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out, name + 'att_pooling_2', is_training)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))
        relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
        return relative_feature

    def pull_and_push_loss(self, val_embedding1, val_embedding2, valid_labels1, valid_labels2, d):

        onehot_valid_labels1 = tf.one_hot(valid_labels1, depth=self.config.num_classes)  # [N1,20]
        onehot_valid_labels2 = tf.one_hot(valid_labels2, depth=self.config.num_classes)  # [N2,20]
        num_class1 = tf.transpose(tf.reduce_sum(onehot_valid_labels1, axis=0, keep_dims=True),
                                  [1, 0])  # [n1,20]——>[20,1]
        num_class2 = tf.transpose(tf.reduce_sum(onehot_valid_labels2, axis=0, keep_dims=True), [1, 0])

        embeding_wise_class1 = tf.matmul(tf.transpose(onehot_valid_labels1, [1, 0]), val_embedding1) / (
                num_class1 + 0.0001)  # [20,k*d]
        embeding_wise_class2 = tf.matmul(tf.transpose(onehot_valid_labels2, [1, 0]), val_embedding2) / (
                num_class2 + 0.0001)

        igter_bool1 = tf.logical_not(tf.equal(tf.squeeze(num_class1, [1]), 0))
        igter_bool2 = tf.logical_not(tf.equal(tf.squeeze(num_class2, [1]), 0))

        indx1 = tf.squeeze(tf.where(tf.equal(igter_bool1, True)), [1])  # [N1]
        indx2 = tf.squeeze(tf.where(tf.equal(igter_bool2, True)), [1])  # [N2]

        and_class = tf.logical_and(igter_bool1, igter_bool2)
        indx_and = tf.squeeze(tf.where(tf.equal(and_class, True)), [1])

        embeding_and1 = tf.gather(embeding_wise_class1, indx_and)
        embeding_and2 = tf.gather(embeding_wise_class2, indx_and)

        embeding_class1 = tf.gather(embeding_wise_class1, indx1)
        embeding_class2 = tf.gather(embeding_wise_class2, indx2)
        and_loss, diff_loss = self.discriminative_push_loss \
            (embeding_class1, embeding_class2, embeding_and1, embeding_and2, indx1, indx2, d, delta_d=2)
        return and_loss, diff_loss

    def discriminative_push_loss(self, embeding_class1, embeding_class2, embeding_and1, embeding_and2, indx1, indx2,
                                    d, delta_d):
        ''' Discriminative loss for a single prediction/label pair.
        :param delta_d: curoff cluster distance
        embeding_class1: [c,d]
        embeding_class1: [c,d]
        '''
        ###### and_loss #####
        mu_and = tf.subtract(embeding_and1, embeding_and2)
        and_loss = tf.nn.l2_loss(mu_and) / tf.cast(d, dtype=tf.float32)

        ##### diff_loss #####
        # Count instances
        mu = tf.concat([embeding_class1, embeding_class2], axis=0)  # [c1+c2=c,d]
        indx = tf.concat([indx1, indx2], axis=0)  # [c]

        num_classes = tf.shape(mu)[0]  # n
        mu_interleaved_rep = tf.tile(mu, [num_classes, 1])
        mu_band_rep = tf.tile(mu, [1, num_classes])
        mu_band_rep = tf.reshape(mu_band_rep, (num_classes * num_classes, d))
        mu_sub = tf.subtract(mu_band_rep, mu_interleaved_rep)

        indx = tf.expand_dims(indx, axis=1)
        indx_interleaved_rep = tf.tile(indx, [num_classes, 1])
        indx_band_rep = tf.tile(indx, [1, num_classes])
        indx_band_rep = tf.reshape(indx_band_rep, (num_classes * num_classes, 1))

        # Filter out zeros from same cluster subtraction
        eye = tf.eye(num_classes)
        zero = tf.zeros(1, dtype=tf.float32)
        diff_cluster_mask = tf.equal(eye, zero)
        diff_cluster_mask = tf.reshape(diff_cluster_mask, [-1])

        mu_sub_bool = tf.boolean_mask(mu_sub, diff_cluster_mask)
        indx_interleaved_rep = tf.boolean_mask(indx_interleaved_rep, diff_cluster_mask)
        indx_band_rep = tf.boolean_mask(indx_band_rep, diff_cluster_mask)

        indx_and_mask = tf.equal(indx_interleaved_rep, indx_band_rep)
        indx_diff_mask = tf.logical_not(indx_and_mask)
        indx_diff_mask = tf.reshape(indx_diff_mask, [-1])
        mu_diff = tf.boolean_mask(mu_sub_bool, indx_diff_mask)

        mu_diff_norm1 = tf.norm(mu_diff, ord=1, axis=1)
        mu_diff_norm2 = tf.subtract(2. * delta_d, mu_diff_norm1)
        mu_diff_norm3 = tf.clip_by_value(mu_diff_norm2, 0., mu_diff_norm2)

        l_dist = tf.reduce_mean(mu_diff_norm3)

        def rt_0(): return 0.

        def rt_l_dist(): return l_dist

        diff_loss = tf.cond(tf.equal(1, num_classes), rt_0, rt_l_dist)

        return and_loss, diff_loss

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        batch_size = tf.shape(interp_idx)[0]
        up_num_points = tf.shape(interp_idx)[1]
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
        interpolated_features = tf.batch_gather(feature, interp_idx)
        interpolated_features = tf.expand_dims(interpolated_features, axis=2)
        return interpolated_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0]
        num_points = tf.shape(pc)[1]
        d = pc.get_shape()[2].value
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
        return features

    @staticmethod
    def att_pooling(feature_set, d_out, name, is_training):
        batch_size = tf.shape(feature_set)[0]
        num_points = tf.shape(feature_set)[1]
        num_neigh = tf.shape(feature_set)[2]
        d = feature_set.get_shape()[3].value
        f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])
        att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name=name + 'fc')
        att_scores = tf.nn.softmax(att_activation, axis=1)
        f_agg = f_reshaped * att_scores
        f_agg = tf.reduce_sum(f_agg, axis=1)
        f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])
        f_agg = helper_tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)
        return f_agg