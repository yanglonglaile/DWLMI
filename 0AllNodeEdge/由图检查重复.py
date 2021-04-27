from numpy import *
import numpy as np
import Tool
import Tool
import random
import math
import os
import time
import pandas as pd
import csv
import math
import random
import networkx as nx

def MyReadCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:          # 注意表头
        SaveList.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

# 数据
# LD
AllNode = []
MyReadCsv(AllNode, "AllNode.csv")
print('AllNode[0]', AllNode[0])


graph = nx.Graph()
print('node')
# 加入所有节点
counter1 = 0
while counter1 < len(AllNode):
    graph.add_node(AllNode[counter1][0])  # 添加一个节点1
    counter1 = counter1 + 1
print('2')
print('图中节点的个数node', graph.number_of_nodes())
print('图中边的个数node', graph.number_of_edges())

AllEdge = []
MyReadCsv(AllEdge, "AllEdge.csv")
print('AllEdge[0]', AllEdge[0])


graph = nx.Graph()
# 加入训练的边
counter1 = 0
while counter1 < len(AllEdge):
    temp = tuple(AllEdge[counter1])
    graph.add_edge(*temp)  # 一次添加一条边
    # print('图中边的个数', graph.number_of_edges())
    # print(counter1)
    counter1 = counter1 + 1

print('3')
print('图中节点的个数', graph.number_of_nodes())
print('图中边的个数', graph.number_of_edges())
print('图中所有的节点', graph.nodes())