# -*- coding: utf-8 -*-
import tensorflow as tf
import config


class Model:
    def __init__(self, vocab_size, num_all_nodes):
        with tf.name_scope('read_inputs') as scope:
            self.all_batch_size = tf.placeholder(tf.int32)
            self.song_batch_size = tf.placeholder(tf.int32)
            self.Text_a = tf.placeholder(tf.int32, [None, config.MAX_LEN], name='Ta')
            self.Text_b = tf.placeholder(tf.int32, [None, config.MAX_LEN], name='Tb')
            self.Text_neg_a = tf.placeholder(tf.int32, [None, config.MAX_LEN], name='Tnega')
            self.Text_neg_b = tf.placeholder(tf.int32, [None, config.MAX_LEN], name='Tnegb')
            self.Text_neg_c = tf.placeholder(tf.int32, [None, config.MAX_LEN], name='Tnegc')

            # node in song set
            self.Text_Node_a = tf.placeholder(tf.int32, [None], name='tn1')
            self.Text_Node_b = tf.placeholder(tf.int32, [None], name='tn2')

            # node in all set，包括song、user、session等
            self.Node_a = tf.placeholder(tf.int32, [None], name='n1')
            self.Node_b = tf.placeholder(tf.int32, [None], name='n2')
            self.Node_neg_a = tf.placeholder(tf.int32, [None], name='nn3')
            self.Node_neg_b = tf.placeholder(tf.int32, [None], name='nn4')
            self.Node_neg_c = tf.placeholder(tf.int32, [None], name='nn5')

            self.all_edge_weight = tf.placeholder(tf.float32, [None], name='aew1')
            self.song_edge_weight = tf.placeholder(tf.float32, [None], name='sew1')

            self.num_batch = tf.placeholder(tf.float32)

        with tf.name_scope('initialize_embedding') as scope:  # text和node的embedding各占一部分
            self.text_embed = tf.Variable(tf.truncated_normal([vocab_size, config.text_embed_size], stddev=0.5))
            self.node_embed = tf.Variable(tf.truncated_normal([num_all_nodes, config.structure_embed_size], stddev=0.5))
            self.harmonious_text_structure_embedding_matrix = tf.Variable(
                tf.truncated_normal([config.text_embed_size, config.structure_embed_size], stddev=0.5))
            self.harmonious_structure_text_embedding_matrix = tf.Variable(
                tf.truncated_normal([config.structure_embed_size, config.text_embed_size], stddev=0.5))

        with tf.name_scope('lookup_embeddings') as scope:
            self.TA = tf.nn.embedding_lookup(self.text_embed, self.Text_a)
            self.T_A = tf.expand_dims(self.TA, -1)

            self.TB = tf.nn.embedding_lookup(self.text_embed, self.Text_b)
            self.T_B = tf.expand_dims(self.TB, -1)

            self.TNEGA = tf.nn.embedding_lookup(self.text_embed, self.Text_neg_a)
            self.T_NEG_A = tf.expand_dims(self.TNEGA, -1)

            self.TNEGB = tf.nn.embedding_lookup(self.text_embed, self.Text_neg_b)
            self.T_NEG_B = tf.expand_dims(self.TNEGB, -1)

            self.TNEGC = tf.nn.embedding_lookup(self.text_embed, self.Text_neg_c)
            self.T_NEG_C = tf.expand_dims(self.TNEGC, -1)

            # 结构的embedding，用于与conv部分交互
            self.T_N_A = tf.nn.embedding_lookup(self.node_embed, self.Text_Node_a)
            self.T_N_B = tf.nn.embedding_lookup(self.node_embed, self.Text_Node_b)

            # 结构的embedding，用于结构信息的交互
            self.N_A = tf.nn.embedding_lookup(self.node_embed, self.Node_a)
            self.N_B = tf.nn.embedding_lookup(self.node_embed, self.Node_b)
            self.N_NEG_A = tf.nn.embedding_lookup(self.node_embed, self.Node_neg_a)
            self.N_NEG_B = tf.nn.embedding_lookup(self.node_embed, self.Node_neg_b)
            self.N_NEG_C = tf.nn.embedding_lookup(self.node_embed, self.Node_neg_c)

        self.convA, self.convB, self.convNegA, self.convNegB, self.convNegC = self.conv()
        self.conv_loss = self.compute_conv_loss()
        self.structure_loss = self.compute_structure_loss()
        h_matrix_weight = 0.1 / (self.num_batch * ((config.embed_size ** 2) / 10000.0))
        structure_parameter_weight = 0.0003 / (config.embed_size / 100.0)
        text_parameter_weight = structure_parameter_weight / 50.0
        self.text_embed_norm = tf.contrib.layers.l2_regularizer(text_parameter_weight)(self.TA) \
                                   + tf.contrib.layers.l2_regularizer(text_parameter_weight)(self.TB)
        self.node_embed_norm = tf.contrib.layers.l2_regularizer(structure_parameter_weight)(self.N_A) \
                                   + tf.contrib.layers.l2_regularizer(structure_parameter_weight)(self.N_B)
        self.matrix_norm = tf.contrib.layers.l2_regularizer(h_matrix_weight)(self.harmonious_text_structure_embedding_matrix) \
                           + tf.contrib.layers.l2_regularizer(h_matrix_weight)(self.harmonious_structure_text_embedding_matrix)
        self.norm_loss = self.text_embed_norm \
                         + self.node_embed_norm \
                         + self.matrix_norm
        self.loss = self.conv_loss \
                    + self.structure_loss \
                    + self.norm_loss

    def conv(self):
        window_size = 3
        W2 = tf.Variable(tf.truncated_normal([window_size, config.text_embed_size, 1, config.text_embed_size], stddev=0.3))

        rand_matrix = tf.Variable(tf.truncated_normal([config.text_embed_size, config.text_embed_size], stddev=0.3))

        convA = tf.nn.conv2d(self.T_A, W2, strides=[1, 1, 1, 1], padding='VALID')
        convB = tf.nn.conv2d(self.T_B, W2, strides=[1, 1, 1, 1], padding='VALID')
        convNEGA = tf.nn.conv2d(self.T_NEG_A, W2, strides=[1, 1, 1, 1], padding='VALID')
        convNEGB = tf.nn.conv2d(self.T_NEG_B, W2, strides=[1, 1, 1, 1], padding='VALID')
        convNEGC = tf.nn.conv2d(self.T_NEG_C, W2, strides=[1, 1, 1, 1], padding='VALID')

        hA = tf.tanh(tf.squeeze(convA))
        hB = tf.tanh(tf.squeeze(convB))
        hNEGA = tf.tanh(tf.squeeze(convNEGA))
        hNEGB = tf.tanh(tf.squeeze(convNEGB))
        hNEGC = tf.tanh(tf.squeeze(convNEGC))

        # config.batch_size --> self.batch_size
        tmphA = tf.reshape(hA, [self.song_batch_size * (config.MAX_LEN - window_size + 1), config.text_embed_size])
        ha_mul_rand = tf.reshape(tf.matmul(tmphA, rand_matrix),
                                 [self.song_batch_size, config.MAX_LEN - window_size + 1, config.text_embed_size])
        r1 = tf.matmul(ha_mul_rand, hB, adjoint_b=True)
        r3 = tf.matmul(ha_mul_rand, hNEGA, adjoint_b=True)
        r4 = tf.matmul(ha_mul_rand, hNEGB, adjoint_b=True)
        r5 = tf.matmul(ha_mul_rand, hNEGC, adjoint_b=True)

        # att1=tf.expand_dims(tf.pack(r1),-1)
        # att3=tf.expand_dims(tf.pack(r3),-1)
        att1 = tf.expand_dims(tf.stack(r1), -1)
        att3 = tf.expand_dims(tf.stack(r3), -1)
        att4 = tf.expand_dims(tf.stack(r4), -1)
        att5 = tf.expand_dims(tf.stack(r5), -1)

        att1 = tf.tanh(att1)
        att3 = tf.tanh(att3)
        att4 = tf.tanh(att4)
        att5 = tf.tanh(att5)
        pooled_A = tf.reduce_mean(att1, 2)
        pooled_B = tf.reduce_mean(att1, 1)
        pooled_NEG_A = tf.reduce_mean(att3, 1)
        pooled_NEG_B = tf.reduce_mean(att4, 1)
        pooled_NEG_C = tf.reduce_mean(att5, 1)

        a_flat = tf.squeeze(pooled_A)
        b_flat = tf.squeeze(pooled_B)
        a_neg_flat = tf.squeeze(pooled_NEG_A)
        b_neg_flat = tf.squeeze(pooled_NEG_B)
        c_neg_flat = tf.squeeze(pooled_NEG_C)

        w_A = tf.nn.softmax(a_flat)
        w_B = tf.nn.softmax(b_flat)
        w_NEG_A = tf.nn.softmax(a_neg_flat)
        w_NEG_B = tf.nn.softmax(b_neg_flat)
        w_NEG_C = tf.nn.softmax(c_neg_flat)

        rep_A = tf.expand_dims(w_A, -1)
        rep_B = tf.expand_dims(w_B, -1)
        rep_NEG_A = tf.expand_dims(w_NEG_A, -1)
        rep_NEG_B = tf.expand_dims(w_NEG_B, -1)
        rep_NEG_C = tf.expand_dims(w_NEG_C, -1)

        hA = tf.transpose(hA, perm=[0, 2, 1])
        hB = tf.transpose(hB, perm=[0, 2, 1])
        hNEGA = tf.transpose(hNEGA, perm=[0, 2, 1])
        hNEGB = tf.transpose(hNEGB, perm=[0, 2, 1])
        hNEGC = tf.transpose(hNEGC, perm=[0, 2, 1])

        rep1 = tf.matmul(hA, rep_A)
        rep2 = tf.matmul(hB, rep_B)
        rep3 = tf.matmul(hNEGA, rep_NEG_A)
        rep4 = tf.matmul(hNEGB, rep_NEG_B)
        rep5 = tf.matmul(hNEGC, rep_NEG_C)

        attA = tf.squeeze(rep1)
        attB = tf.squeeze(rep2)
        attNEGA = tf.squeeze(rep3)
        attNEGB = tf.squeeze(rep4)
        attNEGC = tf.squeeze(rep5)

        return attA, attB, attNEGA, attNEGB, attNEGC

    def compute_conv_loss(self):
        p1 = tf.reduce_sum(tf.multiply(self.convA, self.convB), 1)
        p1 = tf.log(tf.sigmoid(p1) + 0.0001)

        p11 = tf.reduce_sum(tf.multiply(self.convA, self.convNegA), 1)
        p11 = tf.log(tf.sigmoid(-p11) + 0.0001)

        p12 = tf.reduce_sum(tf.multiply(self.convA, self.convNegB), 1)
        p12 = tf.log(tf.sigmoid(-p12) + 0.0001)

        p13 = tf.reduce_sum(tf.multiply(self.convA, self.convNegC), 1)
        p13 = tf.log(tf.sigmoid(-p13) + 0.0001)

        p2 = tf.reduce_sum(
            tf.multiply(tf.matmul(self.convA, self.harmonious_text_structure_embedding_matrix), self.T_N_B), 1)
        p2 = tf.log(tf.sigmoid(p2) + 0.0001)

        p3 = tf.reduce_sum(
            tf.multiply(tf.matmul(self.T_N_A, self.harmonious_structure_text_embedding_matrix), self.convB), 1)
        p3 = tf.log(tf.sigmoid(p3) + 0.0001)

        rho1 = 0.33
        rho3 = 0.11
        temp_loss = rho1 * (p1 + (p11 + p12 + p13)) + rho3 * (p2 + p3)
        loss = - tf.reduce_sum(tf.multiply(temp_loss, self.song_edge_weight))
        return loss

    def compute_structure_loss(self):
        p1 = tf.reduce_sum(tf.multiply(self.N_A, self.N_B), 1)
        p1 = tf.log(tf.sigmoid(p1) + 0.0001)

        p11 = tf.reduce_sum(tf.multiply(self.N_A, self.N_NEG_A), 1)
        p11 = tf.log(tf.sigmoid(-p11) + 0.0001)

        p12 = tf.reduce_sum(tf.multiply(self.N_A, self.N_NEG_B), 1)
        p12 = tf.log(tf.sigmoid(-p12) + 0.0001)

        p13 = tf.reduce_sum(tf.multiply(self.N_A, self.N_NEG_C), 1)
        p13 = tf.log(tf.sigmoid(-p13) + 0.0001)

        temp_loss = p1 + (p11 + p12 + p13)
        loss = - tf.reduce_sum(tf.multiply(temp_loss, self.all_edge_weight))
        return loss
