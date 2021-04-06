#!/usr/bin/env python3

'''
Without intent
'''

import rospkg
import numpy as np
import rospy
import networkx as nx
from mrpp_sumo.srv import NextTaskBot, NextTaskBotResponse, AlgoReady, AlgoReadyResponse
from mrpp_sumo.msg import AtNode
import random as rn
import sys
import os
from numpy.random import default_rng
rng = default_rng()
class CR:

    def __init__(self, g, num_bots):

        self.dirname = rospkg.RosPack().get_path('mrpp_sumo')
        self.name = rospy.get_param("/random_string")
        self.sim_dir = self.dirname + '/post_process/with_intent/' + self.name
        os.mkdir(self.sim_dir)

        self.ready = False
        self.graph = g
        self.stamp = 0.
        self.num_bots = num_bots
        self.no_of_deads = rospy.get_param("/no_of_deads")
        self.nodes = list(self.graph.nodes())
        self.dead_nodes = rng.choice(self.nodes,self.no_of_deads,replace=False)

        # Variable for storing data in sheets
        self.data_arr = np.zeros([1,len(self.nodes)])
        self.global_idle = np.zeros(len(self.nodes))
        self.stamps = np.zeros(1) 

        self.network_arr = {}   #For storing data of all IoT devices

        for i in self.nodes:
            self.network_arr['node_{}'.format(i)] = {}
            for n in self.nodes:
                self.network_arr['node_{}'.format(i)][n] = 0.
        rospy.Service('algo_ready', AlgoReady, self.callback_ready)
        self.ready = True
        

    def callback_idle(self, data):
        if self.stamp < data.stamp:
            dev = data.stamp - self.stamp
            self.stamp = data.stamp

            for n in self.nodes:
                for i in self.nodes:
                    self.network_arr['node_{}'.format(n)][i] += dev

            for n in data.node_id:
                node_index = self.nodes.index(n)
                self.global_idle[node_index] = 0
                
                if n not in self.dead_nodes:
                    self.network_arr['node_{}'.format(n)][n] = 0
                    for neigh_node in list(self.graph.predecessors(n)) :
                        if neigh_node not in self.dead_nodes:
                            self.network_arr['node_{}'.format(neigh_node)][n] = 0.

            self.global_idle +=dev
            self.stamps = np.append(self.stamps,self.stamp)
            self.data_arr = np.append(self.data_arr,[self.global_idle],axis=0)
            
    
    def callback_next_task(self, req):
        node = req.node_done
        t = req.stamp
        bot = req.name
        neigh = list(self.graph.successors(node))
        idles = []

        if node not in self.dead_nodes:
            # print(node,self.dead_nodes)
            for n in neigh:
                idles.append(self.network_arr['node_{}'.format(node)][n])
        else:
            idles = [1 for i in range(len(neigh))]
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

    def save_data(self):
        print("Saving data")
        np.save(self.sim_dir+"/data.npy",self.data_arr)
        np.save(self.sim_dir+"/dead_nodes.npy",self.dead_nodes)
        np.save(self.sim_dir+"/stamps.npy",self.stamps)
        np.save(self.sim_dir+"/nodes.npy",np.array(self.nodes))
        print("Data saved!")


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
    s.save_data()