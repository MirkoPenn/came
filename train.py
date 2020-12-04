import numpy as np
import tensorflow as tf
from DataSet import dataSet
import config
import came
import os
import logging
import psutil
import random

FORMAT = "%(asctime)s - %(message)s"
logging.basicConfig(level=logging.INFO)

# load data
all_graph_path = 'all_graph.txt'
song_graph_path = 'song_graph.txt'
text_path = 'data_all.txt'

logging.info('start reading data.......')
data = dataSet(text_path, all_graph_path, song_graph_path)
logging.info('end reading data.......')

# assign GPU
os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES

# GPU usage amount
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
with tf.Graph().as_default():
    sess = tf.Session(config=gpu_config)
    with sess.as_default():
        model = came.Model(data.num_vocab, data.num_all_nodes)
        opt = tf.train.AdamOptimizer(config.lr)
        train_op = opt.minimize(model.loss)
        sess.run(tf.global_variables_initializer())

        p1 = psutil.Process(os.getpid())

        logging.info('start training.......')
        last_loss_epoch = 0

        for epoch in range(config.num_epoch):
            loss_epoch = 0
            conv_loss_epoch = 0
            structure_loss_epoch = 0
            norm_loss_epoch = 0
            text_norm_loss_epoch = 0
            node_norm_loss_epoch = 0
            matrix_norm_loss_epoch = 0

            if epoch != 0 and epoch % 5 == 0:
                config.all_batch_size += config.init_batch_size
                config.song_batch_size += config.init_batch_size / 2
                logging.info('config.batch_size changes to : {}'.format(config.all_batch_size))

            all_batches = data.generate_all_batches()
            song_batches = data.generate_song_batches()
            all_num_batch = len(all_batches)
            song_num_batch = len(song_batches)
            logging.info('-------------')
            logging.info('embed_size is: {}'.format(config.embed_size))
            logging.info('epoch is: {}'.format(epoch + 1))
            num_batch = max(all_num_batch, song_num_batch)
            for batch_index in range(num_batch):
                if batch_index < len(all_batches):
                    all_batch = all_batches[batch_index]
                else:
                    index = random.randint(0, len(all_batches) - 1)
                    all_batch = all_batches[index]

                if batch_index < len(song_batches):
                    song_batch = song_batches[batch_index]
                else:
                    index = random.randint(0, len(song_batches) - 1)
                    song_batch = song_batches[index]

                node1, node2, neg_node1, neg_node2, neg_node3, all_weight = zip(*all_batch)
                node1, node2, neg_node1, neg_node2, neg_node3, all_weight = np.array(node1), np.array(node2), np.array(
                    neg_node1), np.array(neg_node2), np.array(neg_node3), np.array(all_weight)

                # text song node
                text_node1, text_node2, text_neg_node1, text_neg_node2, text_neg_node3, song_weight = zip(*song_batch)
                text_node1, text_node2, text_neg_node1, text_neg_node2, text_neg_node3, song_weight = np.array(
                    text_node1), np.array(text_node2), np.array(text_neg_node1), np.array(text_neg_node2), np.array(
                    text_neg_node3), np.array(song_weight)
                text1, text2, neg_text1, neg_text2, neg_text3 = data.text[text_node1], data.text[text_node2], data.text[
                    text_neg_node1], data.text[text_neg_node2], data.text[text_neg_node3]

                feed_dict = {
                    # lr: learning_rate,
                    model.song_batch_size: config.song_batch_size,

                    model.Text_a: text1,
                    model.Text_b: text2,
                    model.Text_neg_a: neg_text1,
                    model.Text_neg_b: neg_text2,
                    model.Text_neg_c: neg_text3,

                    model.Text_Node_a: text_node1,
                    model.Text_Node_b: text_node2,

                    model.song_edge_weight: song_weight,

                    model.all_batch_size: config.all_batch_size,

                    model.Node_a: node1,
                    model.Node_b: node2,
                    model.Node_neg_a: neg_node1,
                    model.Node_neg_b: neg_node2,
                    model.Node_neg_c: neg_node3,

                    model.all_edge_weight: all_weight,

                    model.num_batch: num_batch
                }

                _, loss_batch, conv_loss_batch, structure_loss_batch, norm_loss_batch, \
                text_norm_loss_batch, node_norm_loss_batch, matrix_norm_loss_batch = sess.run(
                    [train_op, model.loss, model.conv_loss, model.structure_loss, model.norm_loss,
                     model.text_embed_norm, model.node_embed_norm, model.matrix_norm], feed_dict=feed_dict)

                loss_epoch += loss_batch
                conv_loss_epoch += conv_loss_batch
                structure_loss_epoch += structure_loss_batch
                norm_loss_epoch += norm_loss_batch
                text_norm_loss_epoch += text_norm_loss_batch
                node_norm_loss_epoch += node_norm_loss_batch
                matrix_norm_loss_epoch += matrix_norm_loss_batch
                if batch_index % (num_batch / 5) == 0:
                    logging.info('batch num: {} of {}, progress: {}'.format(batch_index, num_batch, (
                            1.0 * batch_index / num_batch)))
            logging.info('epoch is: {}, and loss is: {}'.format(epoch + 1, loss_epoch))
            logging.info('epoch is: {}, and conv_loss is: {}'.format(epoch + 1, conv_loss_epoch))
            logging.info('epoch is: {}, and structure_loss is: {}'.format(epoch + 1, structure_loss_epoch))
            logging.info('epoch is: {}, and norm_loss_epoch is: {}'.format(epoch + 1, norm_loss_epoch))
            logging.info('epoch is: {}, and text_norm_loss_epoch is: {}'.format(epoch + 1, text_norm_loss_epoch))
            logging.info('epoch is: {}, and node_norm_loss_epoch is: {}'.format(epoch + 1, node_norm_loss_epoch))
            logging.info('epoch is: {}, and matrix_norm_loss_epoch is: {}'.format(epoch + 1, matrix_norm_loss_epoch))
            if last_loss_epoch is not 0:
                change_amount = (loss_epoch - last_loss_epoch) / last_loss_epoch
                logging.info('loss_epoch change amount is: {}'.format(change_amount))
            last_loss_epoch = loss_epoch

            logging.info("memory usage percent: %.2f%%" % (p1.memory_percent()))

        logging.info('start embeding.......')
        logging.info("memory usage percent: %.2f%%" % (p1.memory_percent()))
        node_embed_file = open('node_embed_' + str(config.num_epoch) + '_' + str(config.embed_size) + '.txt', 'wb')
        song_text_embed_file = open('song_text_embed_' + str(config.num_epoch) + '_' + str(config.embed_size) + '.txt',
                                    'wb')

        all_batches = data.generate_all_batches(mode='add')
        song_batches = data.generate_song_batches(mode='add')
        all_num_batch = len(all_batches)
        song_num_batch = len(song_batches)

        node_embed = np.zeros((data.num_all_nodes, config.structure_embed_size))
        song_text_embed = np.zeros((data.num_song_nodes, config.text_embed_size))
        node_weight_sum = np.zeros(data.num_all_nodes)
        song_text_weight_sum = np.zeros(data.num_song_nodes)
        num_batch = max(all_num_batch, song_num_batch)
        for batch_index in range(num_batch):

            if batch_index < len(all_batches):
                all_batch = all_batches[batch_index]
            else:
                index = random.randint(0, len(all_batches) - 1)
                all_batch = all_batches[index]
            if batch_index < len(song_batches):
                song_batch = song_batches[batch_index]
            else:
                index = random.randint(0, len(song_batches) - 1)
                song_batch = song_batches[index]

            node1, node2, neg_node1, neg_node2, neg_node3, all_weight = zip(*all_batch)
            node1, node2, neg_node1, neg_node2, neg_node3, all_weight = np.array(node1), np.array(node2), np.array(
                neg_node1), np.array(neg_node2), np.array(neg_node3), np.array(all_weight)

            # text song node
            text_node1, text_node2, text_neg_node1, text_neg_node2, text_neg_node3, song_weight = zip(*song_batch)
            text_node1, text_node2, text_neg_node1, text_neg_node2, text_neg_node3, song_weight = np.array(
                text_node1), np.array(text_node2), np.array(text_neg_node1), np.array(text_neg_node2), np.array(
                text_neg_node3), np.array(song_weight)
            text1, text2, neg_text1, neg_text2, neg_text3 = data.text[text_node1], data.text[text_node2], data.text[
                text_neg_node1], data.text[text_neg_node2], data.text[text_neg_node3]

            feed_dict = {
                model.song_batch_size: config.song_batch_size,

                model.Text_a: text1,
                model.Text_b: text2,
                model.Text_neg_a: neg_text1,
                model.Text_neg_b: neg_text2,
                model.Text_neg_c: neg_text3,

                model.Text_Node_a: text_node1,
                model.Text_Node_b: text_node2,

                model.song_edge_weight: song_weight,

                model.all_batch_size: config.all_batch_size,

                model.Node_a: node1,
                model.Node_b: node2,
                model.Node_neg_a: neg_node1,
                model.Node_neg_b: neg_node2,
                model.Node_neg_c: neg_node3,

                model.all_edge_weight: all_weight,

                model.num_batch: num_batch

            }

            convA, convB, NA, NB = sess.run([model.convA, model.convB, model.N_A, model.N_B], feed_dict=feed_dict)

            for data_index in range(config.all_batch_size):
                temp_all_weight = all_weight[data_index]
                em = NA[data_index]
                node_embed[node1[data_index]] = node_embed[node1[data_index]] + em * temp_all_weight
                node_weight_sum[node1[data_index]] += temp_all_weight
                em = NB[data_index]
                node_embed[node2[data_index]] = node_embed[node2[data_index]] + em * temp_all_weight
                node_weight_sum[node2[data_index]] += temp_all_weight
            for data_index in range(config.song_batch_size):
                temp_song_weight = song_weight[data_index]
                em = convA[data_index]
                song_text_embed[text_node1[data_index]] = song_text_embed[
                                                              text_node1[data_index]] + em * temp_song_weight
                song_text_weight_sum[text_node1[data_index]] += temp_song_weight
                em = convB[data_index]
                song_text_embed[text_node2[data_index]] = song_text_embed[
                                                              text_node2[data_index]] + em * temp_song_weight
                song_text_weight_sum[text_node2[data_index]] += temp_song_weight

        logging.info('start writing embeding.......')
        logging.info("memory usage percent: %.2f%%" % (p1.memory_percent()))
        for node_index in range(data.num_all_nodes):
            if node_weight_sum[node_index]:
                tmp = node_embed[node_index] / node_weight_sum[node_index]
                node_embed_file.write(' '.join(map(lambda x: format(x, ".6f"), tmp)) + '\n')
            else:
                node_embed_file.write('\n')
        node_embed_file.close()

        for node_index in range(data.num_song_nodes):
            if song_text_weight_sum[node_index]:
                tmp = song_text_embed[node_index] / song_text_weight_sum[node_index]
                song_text_embed_file.write(' '.join(map(lambda x: format(x, ".6f"), tmp)) + '\n')
            else:
                song_text_embed_file.write('\n')
        song_text_embed_file.close()

        logging.info('end writing embeding.......')
        logging.info("memory usage percent: %.2f%%" % (p1.memory_percent()))
