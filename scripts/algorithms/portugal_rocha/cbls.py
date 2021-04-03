#!/usr/bin/env python3

'''
CBLS strategy - Portugal and Rocha
'''

import rospkg
import numpy as np
import rospy
import networkx as nx
from mrpp_sumo.srv import NextTaskBot, NextTaskBotResponse, AlgoReady, AlgoReadyResponse
from mrpp_sumo.msg import AtNode 
import random as rn

class CBLS:

    def __init__(self, g, algo_name, file_path):
        self.ready = False
        self.robots = {}

        #Hyper-parameters
        self.k = 1.
        self.eps = 1e-5
        self.alpha_dur = 0.1
        self.v_max = 10.
        self.stamp = 0.

        self.graph = g
        self.edges = list(self.graph.edges())
        for node in self.graph.nodes():
            self.graph.nodes[node]['idleness'] = 0.
            self.graph.nodes[node]['avg_idle'] = 0.
            self.graph.nodes[node]['future_visits'] = {}
            self.graph.nodes[node]['num_visits'] = 0
        for edge in self.graph.edges():
            self.graph[edge[0]][edge[1]]['duration'] = self.graph[edge[0]][edge[1]]['length']/self.v_max
            # self.graph[edge[0]][edge[1]]['arc_strength'] = self.k
        
        rospy.Service('algo_ready', AlgoReady, self.callback_ready)
        self.ready = True

    def callback_ready(self, req):
        algo_name = req.algo
        if algo_name == 'cbls' and self.ready:
            return AlgoReadyResponse(True)
        else:
            return AlgoReadyResponse(False)

    def callback_idle(self, data):
        if self.stamp < data.stamp:
            dev = data.stamp - self.stamp
            self.stamp = data.stamp
            for i in self.graph.nodes():
                self.graph.nodes[i]['idleness'] += dev
                for j in self.graph.nodes[i]['future_visits'].keys():
                    self.graph.nodes[i]['future_visits'][j] -= dev
                self.graph.nodes[i]['avg_idle'] = (self.graph.nodes[i]['avg_idle'] *(self.stamp - dev) + (self.graph.nodes[i]['idleness'] - dev/2) * dev)/self.stamp

            for i, n in enumerate(data.node_id):
                self.graph.nodes[n]['idleness'] = 0.
                self.graph.nodes[n]['num_visits'] += 1
                if data.robot_id[i] in self.graph.nodes[n]['future_visits'].keys():
                    if self.graph.nodes[n]['future_visits'][data.robot_id[i]] < 0.:
                        self.graph.nodes[n]['future_visits'].pop(data.robot_id[i])

            for i, r in enumerate(data.robot_id):
                if r not in self.robots.keys():
                    self.robots[r] = {}
                    self.robots[r]['last_node'] = data.node_id[i]
                    self.robots[r]['last_visit'] = self.stamp
                    self.robots[r]['arc_strengths'] = []
                    for edge in self.edges:
                        self.robots[r]['arc_strengths'].append(self.k)

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
        # prob = np.random.random_sample()
        print ('Time {}, Bot {}, Node {}'.format(t, bot, node))
        priors = []
        neigh = list(self.graph.successors(node))
        sum_idle = 0                    
        conds = []
        prob_move = []
        chis = [] 
        if bot not in self.robots.keys():
            self.robots[bot] = {}
            self.robots[bot]['last_node'] = node
            self.robots[bot]['last_visit'] = t
            self.robots[bot]['arc_strengths'] = []
            for edge in self.edges:
                self.robots[bot]['arc_strengths'].append(self.k)
        sum_strength = np.sum(self.robots[bot]['arc_strengths'])
        for i in list(self.graph.nodes()):
            sum_idle += self.graph.nodes[i]['idleness']

        for n in neigh:
            if sum_idle > 0.:
                priors.append(self.graph.nodes[n]['idleness']/sum_idle)
            else:
                priors.append(1./len(neigh))
            edge_ind = self.edges.index((node, n))
            conds.append(self.robots[bot]['arc_strengths'][edge_ind]/sum_strength)
            prob_move.append(priors[-1] * conds[-1] + self.eps)

        prob_move /= np.sum(prob_move)
        prob_move = list(prob_move)
        # print ('neigh', len(neigh), 'prob_move', prob_move)
        if len(neigh) > 1:
            ent = - np.sum(prob_move * np.log2(prob_move)) / np.log2(len(neigh))
            max_ids = list(np.where(prob_move == np.amax(prob_move))[0])
            max_id = rn.sample(max_ids, 1)[0]
        else:
            ent = 0.
            max_id = 0
        # print ('ent', ent)

        next_node = neigh[max_id]

        temp1 = prob_move.copy()
        temp2 = neigh.copy()
        while len(self.graph.nodes[next_node]['future_visits'].keys()) > 0:
            if len(temp1) == 1:
                break
            temp2.remove(next_node)
            # max_ids.pop(0)
            temp1.pop(max_id)
            # if len(max_ids) > 0:
            #     max_id = rn.sample(max_ids, 1)[0]
            #     next_node = temp2[max_id]
            # else:
            #     print (temp1)
            max_ids = list(np.where(temp1 == np.amax(temp1))[0])
            max_id = rn.sample(max_ids, 1)[0]
            next_node = temp2[max_id]
                
        for n in neigh:
            temp = self.graph.nodes[n]['num_visits']
            if n == next_node:
                temp += 1            
            chis.append(temp/len(list(self.graph.predecessors(n))))

        pri_ids1 = list(np.where(priors == np.amax(priors))[0])
        pri_ids2 = list(np.where(priors == np.amin(priors))[0])
        chi_ids1 = list(np.where(chis == np.amax(chis))[0])
        chi_ids2 = list(np.where(chis == np.amin(chis))[0])

        for n in neigh:
            edge_ind = self.edges.index((node, n))
            s = 0.
            if len(neigh) > 1:
                if n in chi_ids1:
                    if (len(chi_ids1) > 1) and (n in pri_ids2):
                        s = -1.
                    elif (len(chi_ids1 == 1)):
                        s = -1.
                if n in chi_ids2:
                    if (len(chi_ids2) > 1) and (n in pri_ids1):
                        s = 1.
                    elif (len(chi_ids2 == 1)):
                        s = 1.
            self.robots[bot]['arc_strengths'][edge_ind] = self.robots[bot]['arc_strengths'][edge_ind] + s*(1 - ent) 
        next_walk = [node, next_node]
        next_departs = [t]
        self.graph.nodes[next_node]['future_visits'][bot] = self.graph[node][next_node]['duration']
        # print (next_walk, next_departs)
        return NextTaskBotResponse(next_departs, next_walk)

if __name__ == '__main__':
    rospy.init_node('cbls', anonymous = True)
    dirname = rospkg.RosPack().get_path('mrpp_sumo')
    done = False
    graph_name = rospy.get_param('/graph')
    g = nx.read_graphml(dirname + '/graph_ml/' + graph_name + '.graphml')

    # priority_nodes = rospy.get_param('/priority_nodes').split(' ')
    # lambda_priority = float(rospy.get_param('/lambda_priority'))
    algo_name = rospy.get_param('/algo_name')
    file_name = rospy.get_param('/random_string')
    file_path = dirname + '/outputs/{}_command.in'.format(file_name)
    s = CBLS(g, algo_name, file_path)

    rospy.Subscriber('at_node', AtNode, s.callback_idle)
    rospy.Service('bot_next_task', NextTaskBot, s.callback_next_task)

    done = False
    while not done:
        done = rospy.get_param('/done')