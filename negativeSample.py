# -*- coding: utf-8 -*-
from math import pow
from config import neg_table_size, NEG_SAMPLE_POWER
import logging

def InitNegTable(edges):
    a_list, b_list, weight_list = zip(*edges)
    a_list = list(a_list)
    b_list = list(b_list)
    weight_list = list(weight_list)

    node_degree = {}
    for i in range(len(a_list)):
        if a_list[i] in node_degree:
            node_degree[a_list[i]] += weight_list[i]
        else:
            node_degree[a_list[i]] = weight_list[i]
    for i in range(len(b_list)):
        if b_list[i] in node_degree:
            node_degree[b_list[i]] += weight_list[i]
        else:
            node_degree[b_list[i]] = weight_list[i]

    sum_degree = 0
    for i in node_degree.values():
        sum_degree += pow(i, 0.75)

    por = 0
    cur_sum = 0
    vid = -1
    neg_table = []
    degree_list = list(node_degree.values())
    node_id = list(node_degree.keys())
    for i in range(neg_table_size):
        if ((i + 1) / float(neg_table_size)) > por:
            cur_sum += pow(degree_list[vid + 1], NEG_SAMPLE_POWER)
            por = cur_sum / sum_degree
            vid += 1
        neg_table.append(node_id[vid])
    return neg_table
