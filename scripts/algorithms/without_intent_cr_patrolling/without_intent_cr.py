#!/usr/bin/env python3

'''
Conscientious Reactive
'''

import rospkg
import numpy as np
import rospy
import networkx as nx
from mrpp_sumo.srv import NextTaskBot, NextTaskBotResponse, AlgoReady, AlgoReadyResponse
from mrpp_sumo.msg import AtNode
import random as rn

class CR:

    def __init__(self, g, num_bots):
        self.ready = False
        self.graph = g
        self.stamp = 0.
        self.num_bots = num_bots
        self.no_of_deads = 5

        self.nodes = list(self.graph.nodes())
        self.dead_nodes = rn.sample(self.nodes,self.no_of_deads)
        print(self.dead_nodes)
        self.network_arr = {}

        for i in self.nodes:
            self.network_arr['node_{}'.format(i)] = {}
            for n in self.nodes:
                self.network_arr['node_{}'.format(i)][n] = 0.
        rospy.Service('algo_ready', AlgoReady, self.callback_ready)
        self.ready = True
        # print(self.network_arr)
        
    def callback_idle(self, data):
        # print(self.network_arr["node_0"][self.dead_nodes[0]])
        if self.stamp < data.stamp:
            dev = data.stamp - self.stamp
            self.stamp = data.stamp

            for n in self.nodes:
                for i in self.nodes:
                    self.network_arr['node_{}'.format(n)][i] += dev
 
    def callback_next_task(self, req):
        node = req.node_done
        t = req.stamp
        bot = req.name
        neigh = list(self.graph.successors(node))
        idles = []
        if node not in self.dead_nodes:
            self.network_arr['node_{}'.format(node)][node] = 0
            for neigh_node in list(self.graph.successors(node)) :
                if neigh_node not in self.dead_nodes:
                    self.network_arr['node_{}'.format(neigh_node)][node] = 0.

            for n in neigh:
                idles.append(self.network_arr['node_{}'.format(node)][n])
        else:
            idles = [1 for i in range(len(neigh))]

        print(node,neigh,idles)

        max_id = 0
        if len(neigh) > 1:
            max_ids = list(np.where(idles == np.amax(idles))[0])
            max_id = rn.sample(max_ids, 1)[0]
        next_walk = [node, neigh[max_id]]
        self.network_arr['node_{}'.format(node)][neigh[max_id]] = 0
        next_departs = [t]
        return NextTaskBotResponse(next_departs, next_walk)

    def callback_ready(self, req):
        algo_name = req.algo
        if algo_name == 'with_intent' and self.ready:
            return AlgoReadyResponse(True)
        else:
            return AlgoReadyResponse(False)

    # def save_sheet():
    #     print("data_saved")

if __name__ == '__main__':
    rospy.init_node('cr', anonymous= True)
    dirname = rospkg.RosPack().get_path('mrpp_sumo')
    done = False
    graph_name = rospy.get_param('/graph')
    num_bots = int(rospy.get_param('/init_bots'))
    g = nx.read_graphml(dirname + '/graph_ml/' + graph_name + '.graphml')

    s = CR(g, num_bots)

    rospy.Subscriber('at_node', AtNode, s.callback_idle)
    rospy.Service('bot_next_task', NextTaskBot, s.callback_next_task)

    done = False
    while not done and not rospy.is_shutdown():
        done = rospy.get_param('/done')