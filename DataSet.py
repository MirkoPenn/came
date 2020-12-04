# -*- coding: utf-8 -*-
import functools
import sys
import operator
import config
import numpy as np
from negativeSample import InitNegTable
import random
import logging


class dataSet:
    def __init__(self, text_path, all_graph_path, song_graph_path):

        text_file, all_graph_file, song_graph_file = self.load(text_path, all_graph_path, song_graph_path)

        self.all_edges = self.load_edges(all_graph_file)
        self.song_edges = self.load_edges(song_graph_file)

        logging.info('len(self.all_edges) is: {}'.format(len(self.all_edges)))
        logging.info('len(self.song_edges) is: {}'.format(len(self.song_edges)))
        self.text, self.num_vocab, self.num_song_nodes = self.load_text(text_file)

        logging.info('len(self.text) is: {}'.format(len(self.text)))

        logging.info('self.num_vocab is: {}'.format(self.num_vocab))
        logging.info('self.num_nodes is: {}'.format(self.num_song_nodes))

        if self.num_song_nodes != 361570:
            for i in range(10):
                logging.info('self.num_song_nodes changes from {} to {}'.format(self.num_song_nodes, 361570))
            # self.num_song_nodes = 361570

        self.all_negative_table = InitNegTable(self.all_edges)
        self.song_negative_table = InitNegTable(self.song_edges)
        logging.info('len(self.all_negative_table) is: {}'.format(len(self.all_negative_table)))
        logging.info('len(self.song_negative_table) is: {}'.format(len(self.song_negative_table)))

        self.all_exist_node1_node2set_dict, self.num_all_nodes = self.get_exist_node1_node2set_dict(self.all_edges)
        self.song_exist_node1_node2set_dict, self.num_song_nodes_temp = self.get_exist_node1_node2set_dict(
            self.song_edges)
        logging.info('self.num_all_nodes is: {}'.format(self.num_all_nodes))
        logging.info('self.num_song_nodes_temp is: {}'.format(self.num_song_nodes_temp))
        # check the node count
        true_all_node_manually = 361570 + 4284 + 165472
        if self.num_all_nodes != true_all_node_manually:
            for i in range(10):
                logging.info(
                    'self.num_all_nodes changes from {} to {}'.format(self.num_all_nodes, true_all_node_manually))
            # self.num_all_nodes = true_all_node_manually

    def load(self, text_path, all_graph_path, song_graph_path):
        text_file = open(text_path, 'r').readlines()
        all_graph_file = open(all_graph_path, 'r').readlines()
        song_graph_file = open(song_graph_path, 'r').readlines()

        return text_file, all_graph_file, song_graph_file

    def load_edges(self, graph_file):
        edges = []
        for i in graph_file:
            edge = i.strip().split(',')
            edges.append(list(map(int, edge[:2])) + list(map(float, edge[2:])))
        return edges  # node1 node2 weight

    def load_text(self, text_file):
        word_freq_dict = {}
        for line in text_file:
            words = line.strip().split(" ")
            for word in words:
                if word in word_freq_dict:
                    word_freq_dict[word] = word_freq_dict[word] + 1
                else:
                    word_freq_dict[word] = 1
        del word_freq_dict['']
        sorted_word_freq_dict = dict(sorted(word_freq_dict.items(), key=lambda item: item[1], reverse=True))
        word_index_dict = {"<UNK>": 0}
        for x in sorted_word_freq_dict:
            if sorted_word_freq_dict[x] == 1:
                break
            else:
                word_index_dict[x] = len(word_index_dict)

        text = np.zeros((len(text_file), config.MAX_LEN), dtype='int32')
        for i in range(len(text_file)):
            words = text_file[i].strip().split(" ")
            j = 0
            for word in words:
                if j == config.MAX_LEN:
                    break
                if word in word_index_dict:
                    index = word_index_dict[word]
                    text[i][j] = index
                    j += 1
        num_vocab = len(word_index_dict)
        num_song_nodes = len(text)

        return text, num_vocab, num_song_nodes

    def negative_sample(self, edges, negative_table, exist_node1_node2set_dict):
        node1, node2, weight = zip(*edges)
        sample_edges = []
        func = lambda: negative_table[random.randint(0, config.neg_table_size - 1)]
        for i in range(len(edges)):
            sample_lower_bound = 0
            sample_upper_bound = self.num_all_nodes
            if node2[i] < self.num_song_nodes:  # 2nd node is song
                sample_upper_bound = self.num_song_nodes
            elif node2[i] >= self.num_song_nodes + 15981:  # 2nd node is session
                sample_lower_bound = self.num_song_nodes + 15981
            neg_node = []
            for j in range(3):
                temp_neg_node = func()
                while temp_neg_node == node1[i] or temp_neg_node == node2[i] or \
                        (temp_neg_node in exist_node1_node2set_dict[node1[i]]) or \
                        temp_neg_node < sample_lower_bound or \
                        temp_neg_node >= sample_upper_bound:
                    temp_neg_node = func()
                neg_node.append(temp_neg_node)
            sample_edges.append([node1[i], node2[i], neg_node[0], neg_node[1], neg_node[2], weight[i]])
        return sample_edges

    def negative_sample_old(self, edges, negative_table, exist_node1_node2set_dict):
        node1, node2, weight = zip(*edges)

        sample_edges = []
        func = lambda: negative_table[random.randint(0, config.neg_table_size - 1)]
        for i in range(len(edges)):
            neg_node = []
            for j in range(3):
                temp_neg_node = func()
                while temp_neg_node == node1[i] or temp_neg_node == node2[i] or (
                        temp_neg_node in exist_node1_node2set_dict[node1[i]]) or temp_neg_node >= self.num_song_nodes:
                    temp_neg_node = func()
                neg_node.append(temp_neg_node)
            sample_edges.append([node1[i], node2[i], neg_node[0], neg_node[1], neg_node[2], weight[i]])
        return sample_edges

    def generate_all_batches(self, mode=None):
        all_num_batch = len(self.all_edges) // config.all_batch_size
        edges = self.all_edges
        if mode == 'add':
            all_num_batch += 1
            edges.extend(edges[:int(config.all_batch_size - len(self.all_edges) % config.all_batch_size)])
        if mode != 'add':
            random.shuffle(edges)
        sample_edges = edges[:int(all_num_batch * config.all_batch_size)]
        sample_edges = self.negative_sample(sample_edges, self.all_negative_table, self.all_exist_node1_node2set_dict)

        batches = []
        for i in range(int(all_num_batch)):
            batches.append(sample_edges[int(i * config.all_batch_size):int((i + 1) * config.all_batch_size)])
        return batches

    def generate_song_batches(self, mode=None):

        song_num_batch = len(self.song_edges) // config.song_batch_size
        edges = self.song_edges
        if mode == 'add':
            song_num_batch += 1
            edges.extend(edges[:int(config.song_batch_size - len(self.song_edges) % config.song_batch_size)])
        if mode != 'add':
            random.shuffle(edges)
        sample_edges = edges[:int(song_num_batch*config.song_batch_size)]
        sample_edges = self.negative_sample(sample_edges, self.song_negative_table, self.song_exist_node1_node2set_dict)

        batches = []
        for i in range(int(song_num_batch)):
            batches.append(sample_edges[int(i * config.song_batch_size):int((i + 1) * config.song_batch_size)])
        return batches

    def get_exist_node1_node2set_dict(self, edges):

        temp_node_set = set()
        node1, node2, weight = zip(*edges)

        exist_node1_node2set_dict = {}
        for i in range(len(edges)):
            temp_node_set.add(node1[i])
            temp_node_set.add(node2[i])
            node2_set = exist_node1_node2set_dict.get(node1[i], set())
            node2_set.add(node2[i])
            exist_node1_node2set_dict[node1[i]] = node2_set
        return exist_node1_node2set_dict, len(temp_node_set)


if __name__ == '__main__':
    all_graph_path = 'all_graph.txt'
    song_graph_path = 'song_graph.txt'
    text_path = 'data_all.txt'
    data = dataSet(text_path, all_graph_path, song_graph_path)
    print("end")
