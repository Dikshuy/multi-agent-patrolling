#!/usr/bin/env python3

'''
SEBS strategy - Portugal and Rocha
'''

import rospkg
import numpy as np
import rospy
import networkx as nx
from mrpp_sumo.srv import NextTaskBot, NextTaskBotResponse, AlgoReady, AlgoReadyResponse
from mrpp_sumo.msg import AtNode 
import random as rn

class SEBS:

    def __init__(self, g, num_robots):
        self.ready = False
        self.robots = {}

        #Hyper-parameters
        self.eps = 1e-5
        self.alpha_dur = 0.1
        self.L = 0.1
        self.M = 1.
        self.v_max = 10.

        self.stamp = 0.
        self.num_robots = num_robots
        self.graph = g
        self.edges = list(self.graph.edges())
        for node in self.graph.nodes():
            self.graph.nodes[node]['idleness'] = 0.
            self.graph.nodes[node]['future_visits'] = {}
        for edge in self.graph.edges():
            self.graph[edge[0]][edge[1]]['duration'] = self.graph[edge[0]][edge[1]]['length']/self.v_max

        rospy.Service('algo_ready', AlgoReady, self.callback_ready)
        self.ready = True

    def callback_ready(self, req):
        algo_name = req.algo
        if algo_name == 'sebs' and self.ready:
            return AlgoReadyResponse(True)
        else:
            return AlgoReadyResponse(False)

    def callback_idle(self, data):
        if self.stamp < data.stamp:
            dev = data.stamp - self.stamp
            self.stamp = data.stamp
            
            for i in self.graph.nodes():
                self.graph.nodes[i]['idleness'] += dev
                if self.graph.nodes[i]['idleness'] > self.M:
                    self.M = self.graph.nodes[i]['idleness']
                    
            for i, n in enumerate(data.node_id):
                self.graph.nodes[n]['idleness'] = 0.
                if data.robot_id[i] in self.graph.nodes[n]['future_visits'].keys():
                    if self.graph.nodes[n]['future_visits'][data.robot_id[i]] < 0.:
                        self.graph.nodes[n]['future_visits'].pop(data.robot_id[i])

            for i, r in enumerate(data.robot_id):
                if r not in self.robots.keys():
                    self.robots[r] = {}
                    self.robots[r]['last_node'] = data.node_id[i]
                    self.robots[r]['last_visit'] = self.stamp

                elif self.robots[r]['last_node'] == data.node_id[i]:
                    self.robots[r]['last_visit'] = self.stamp

                else:
                    last_delta = self.stamp - self.robots[r]['last_visit']
                    self.graph[self.robots[r]['last_node']][data.node_id[i]]['duration'] = (1 - self.alpha_dur) * self.graph[self.robots[r]['last_node']][data.node_id[i]]['duration'] + self.alpha_dur * last_delta
                    self.robots[r]['last_node'] = data.node_id[i]
                    self.robots[r]['last_visit'] = self.stamp


    def callback_next_task(self, req):
        node = req.node_done
        t = req.stamp
        bot = req.name

        # print ('Time {}, Bot {}, Node {}'.format(t, bot, node))
        neigh = list(self.graph.successors(node))
        if bot not in self.robots.keys():
            self.robots[bot] = {}
            self.robots[bot]['last_node'] = node
            self.robots[bot]['last_visit'] = t

        p_g = []
        for n in neigh:
            g = self.graph.nodes[n]['idleness']/self.graph[node][n]['duration']
            g_1 = self.L * np.exp(g * np.log(1/self.L) / self.M)
            s = len(self.graph.nodes[n]['future_visits'].keys())
            s_1 = 2 ** (self.num_robots - s - 1)/(2 ** self.num_robots - 1)
            p_g.append(g_1 * s_1)

        if len(neigh) > 1:
            max_ids = list(np.where(p_g == np.amax(p_g))[0])
            max_id = int(rn.sample(max_ids, 1)[0])
            next_node = neigh[max_id]
        else:
            next_node = neigh[0]
        next_walk = [node, next_node]
        next_departs = [t]
        self.graph.nodes[next_node]['future_visits'][bot] = self.graph[node][next_node]['duration']
        # print (next_walk, next_departs)
        return NextTaskBotResponse(next_departs, next_walk)

if __name__ == '__main__':
    rospy.init_node('sebs', anonymous = True)
    dirname = rospkg.RosPack().get_path('mrpp_sumo')
    done = False
    graph_name = rospy.get_param('/graph')
    g = nx.read_graphml(dirname + '/graph_ml/' + graph_name + '.graphml')
    num_robots = int(rospy.get_param('/init_bots'))
    s = SEBS(g, num_robots)
    rospy.Subscriber('at_node', AtNode, s.callback_idle)
    rospy.Service('bot_next_task', NextTaskBot, s.callback_next_task)

    done = False
    while not done:
        done = rospy.get_param('/done')