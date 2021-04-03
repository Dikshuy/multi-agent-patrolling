#!/usr/bin/env python3

'''
Reactive with Flags
'''

import rospkg
import numpy as np
import rospy
import networkx as nx
from mrpp_sumo.srv import NextTaskBot, NextTaskBotResponse, AlgoReady, AlgoReadyResponse
from mrpp_sumo.msg import AtNode
import random as rn

class RR:

    def __init__(self, g):
        self.ready = False
        self.graph = g
        self.stamp = 0.

        for node in self.graph.nodes():
            self.graph.nodes[node]['idleness'] = 0.
        rospy.Service('algo_ready', AlgoReady, self.callback_ready)
        self.ready = True

    def callback_idle(self, data):
        if self.stamp < data.stamp:
            dev = data.stamp - self.stamp
            self.stamp = data.stamp
            for i in self.graph.nodes():
                self.graph.nodes[i]['idleness'] += dev
                
            for i, n in enumerate(data.node_id):
                self.graph.nodes[n]['idleness'] = 0.
                
    
    def callback_next_task(self, req):
        node = req.node_done
        t = req.stamp

        self.graph.nodes[node]['idleness'] = 0.

        neigh = list(self.graph.successors(node))
        idles = []
        for n in neigh:
            idles.append(self.graph.nodes[n]['idleness'])

        max_id = 0
        if len(neigh) > 1:
            max_ids = list(np.where(idles == np.amax(idles))[0])
            max_id = rn.sample(max_ids, 1)[0]
        next_walk = [node, neigh[max_id]]
        next_departs = [t]
        return NextTaskBotResponse(next_departs, next_walk)

    def callback_ready(self, req):
        algo_name = req.algo
        if algo_name == 'reactive_with_flags' and self.ready:
            return AlgoReadyResponse(True)
        else:
            return AlgoReadyResponse(False)

if __name__ == '__main__':
    rospy.init_node('rr', anonymous= True)
    dirname = rospkg.RosPack().get_path('mrpp_sumo')
    done = False
    graph_name = rospy.get_param('/graph')
    g = nx.read_graphml(dirname + '/graph_ml/' + graph_name + '.graphml')

    s = RR(g)

    rospy.Subscriber('at_node', AtNode, s.callback_idle)
    rospy.Service('bot_next_task', NextTaskBot, s.callback_next_task)

    done = False
    while not done:
        done = rospy.get_param('/done')