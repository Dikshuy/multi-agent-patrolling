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

        self.nodes = list(self.graph.nodes())
        self.robots = {}
        for i in range(num_bots):
            self.robots['bot_{}'.format(i)] = {}
            for n in self.nodes:
                self.robots['bot_{}'.format(i)][n] = 0.

        rospy.Service('algo_ready', AlgoReady, self.callback_ready)
        self.ready = True

    def callback_idle(self, data):
        if self.stamp < data.stamp:
            dev = data.stamp - self.stamp
            self.stamp = data.stamp
            for n in range(num_bots):
                for i in self.nodes:
                    self.robots['bot_{}'.format(n)][i] += dev
                
            for i, n in enumerate(data.robot_id):
                self.robots[n][data.node_id[i]] = 0.
                
    
    def callback_next_task(self, req):
        node = req.node_done
        t = req.stamp
        bot = req.name
        
        self.robots[bot][node] = 0.

        neigh = list(self.graph.successors(node))
        idles = []
        for n in neigh:
            idles.append(self.robots[bot][n])

        max_id = 0
        if len(neigh) > 1:
            max_ids = list(np.where(idles == np.amax(idles))[0])
            max_id = rn.sample(max_ids, 1)[0]
        next_walk = [node, neigh[max_id]]
        next_departs = [t]
        return NextTaskBotResponse(next_departs, next_walk)

    def callback_ready(self, req):
        algo_name = req.algo
        if algo_name == 'conscientious_reactive' and self.ready:
            return AlgoReadyResponse(True)
        else:
            return AlgoReadyResponse(False)

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
    while not done:
        done = rospy.get_param('/done')