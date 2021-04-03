#!/usr/bin/env python3


'''
Parameters:
priority_nodes
lambda - weight of the priority nodes
time_period - time period of the priority nodes
plan_time - the time for which path needs to be planned
'''

import rospkg
import numpy as np
import rospy
import networkx as nx
import configparser as CP
from mrpp_sumo.srv import NextTaskBot, NextTaskBotResponse
from mrpp_sumo.msg import AtNode 
import tpbp_functions as func_tp

class TPBP_Basic:
    def __init__(self, g, priority_nodes, time_periods, lambda_priority, length_walk, max_div, eps_prob, discount_factor, algo_name):

        self.robots = {}

        #Hyper-parameters:
        self.alpha_dur = 0.1
        self.v_max = 10.
        self.num_samples = 5

        #Graph Initializations
        self.graph = g
        for node in self.graph.nodes():
            self.graph.nodes[node]['idleness'] = 0.
            self.graph.nodes[node]['future_visits'] = {}
        for edge in self.graph.edges():
            self.graph[edge[0]][edge[1]]['duration'] = self.graph[edge[0]][edge[1]]['length']/self.v_max

        self.priority_nodes = priority_nodes
        self.time_periods = time_periods
        self.plan_time = length_walk   
        self.lambda_priority = lambda_priority
        self.max_div = max_div
        self.eps_prob = eps_prob
        self.cf = self.plan_time/self.max_div
        self.discount_factor = discount_factor
        self.algo_name = algo_name
        self.stamp = 0.0
        
        self.assigned = []
        for _ in self.priority_nodes:
            self.assigned.append(False)
        self.robot_cur_walks = {}
        self.robot_pub = []

        self.nodes = list(self.graph.nodes())
        self.N = len(self.nodes)

        #Initializing Constant Arrays
        self.t_int = np.zeros([self.N, self.N])
        self.t_div = np.zeros([self.N, self.N])
        self.adj_mat_id = -1 * np.ones([self.N, self.N, self.N, self.max_div + 1], dtype = 'int64')
        self.update_adj_matrix('')

    def update_adj_matrix(self, event):

        for i in range(self.N):
            orig = self.nodes[i]
            (ta, _) = func_tp.shortest_path_one_to_all(self.graph, orig, 'duration')
            for k in range(self.N):
                self.t_int[i][k] = int(np.ceil(ta[self.nodes[k]]))
                self.t_div[i][k] = int(np.ceil(self.t_int[i][k] / self.cf))

            for l in range(self.N):
                for m in range(self.N):
                    if (self.nodes[l], self.nodes[m]) in self.graph.edges():
                        t = self.graph[self.nodes[l]][self.nodes[m]]['duration']
                        tdiv = int(np.floor(t / self.cf))
                        for n in range(self.max_div + 1):
                            if (n >= self.t_div[i][m]) and ((n - tdiv) >= self.t_div[i][l]):
                                self.adj_mat_id[i, l, m, n] = int(n - tdiv)
                    elif l == m:
                        for n in range(self.max_div + 1):
                            if n > self.t_div[i][m]:
                                self.adj_mat_id[i, l, m, n] = int(n - 1)
            self.adj_mat_id[i, i, i, 0] = 0

    def monte_carlo_sampling(self, orig, len_cur, dest = []):
        
        g_walk = []
        g_max = 0.

        orig_id = self.nodes.index(orig)
        adj_temp = self.adj_mat_id[orig_id]
        div_cur = int(np.floor_divide(len_cur, self.cf))

        vals_n = np.zeros([self.N, div_cur + 1, self.N, div_cur + 1])
        node_val = np.zeros(self.N)
        future_visit_final = np.zeros(self.N)
        for i, w in enumerate(self.nodes):
            node_val[i] = self.graph.nodes[w]['idleness']
            if len(self.graph.nodes[w]['future_visits'].values()) > 0:
                future_visit_final[i] = max(self.graph.nodes[w]['future_visits'].values())
            else:
                future_visit_final[i] = -1. * node_val[i]

            #Node Weight Function
            for m in range(div_cur + 1):
                if m >= self.t_div[orig_id, i]:
                    vals_n[orig_id, 0, i, m] = (self.discount_factor ** m) * max(m * self.cf - future_visit_final[i], 0)
                    if w in self.priority_nodes and max(m * self.cf - future_visit_final[i], 0) <= self.time_periods[self.priority_nodes.index(w)]:
                        vals_n[orig_id, 0, i, m] += self.lambda_priority * max(m * self.cf - future_visit_final[i], 0)

        if min(future_visit_final) > max(self.time_periods):
            return g_walk, g_max

        for _ in range(self.num_samples):
            w, g = self.determine_walk(adj_temp, orig, div_cur, vals_n[:], future_visit_final, dest)
            if g > g_max:
                g_walk = w[:]
                g_max = g
        return g_walk, g_max


    def determine_walk(self, adj_temp, orig, div_cur, vals_n, future_visit_final, dest = []):      

        g_n = np.zeros([self.N, div_cur + 1])

        walks = []
        for i in range(self.N):
            walks.append([])
            for j in range(div_cur + 1):
                walks[i].append([])
                 
        for i in range(div_cur + 1):
            for j in range(self.N):
                g_e = -1. * np.ones(self.N)
                for k in range(self.N):
                    if adj_temp[k, j, i] >= 0:
                        g_e[k] = g_n[k, adj_temp[k, j, i]] + vals_n[k, adj_temp[k, j, i], j, i]
                max_g = np.max(g_e)
                if max_g >= 0.:
                    args_temp = np.argwhere(g_e == max_g).flatten().tolist()

                    #Random Selection    
                    sel_k = np.random.choice(args_temp)
                    vals_l = np.copy(vals_n[sel_k, adj_temp[sel_k, j, i]])

                    #Node Weight Update 
                    if i <= div_cur and i * self.cf > future_visit_final[j]:
                        for n in range(i, div_cur + 1):
                            vals_l[j, n] = (self.discount_factor ** n) * (n - i) * self.cf 
                            if self.nodes[j] in self.priority_nodes:
                                if (n - i) * self.cf <= self.time_periods[self.priority_nodes.index(self.nodes[j])]:
                                    vals_l[j, n] += self.lambda_priority * (n - i) * self.cf

                    vals_n[j, i] = vals_l
                    g_n[j, i] = max_g
                    walks[j][i] = walks[sel_k][adj_temp[sel_k, j, i]][:]
                    walks[j][i].append(j)

        if dest == []:
            dest = self.nodes
        dest_ids = []
        for n in dest:
            dest_ids.append(self.nodes.index(n))
        max_g = np.max(g_n[dest_ids, div_cur])
        args_temp = np.argwhere(g_n[dest_ids, div_cur] == max_g).flatten().tolist()

        #Random Selection
        sel_dest = dest_ids[np.random.choice(args_temp)]
        walk_id = walks[sel_dest][div_cur]
        walk = []
        for i in walk_id:
            walk.append(self.nodes[i])
        return walk, max_g

    def callback_idle(self, data):
        if self.stamp < data.stamp:
            dev = data.stamp - self.stamp
            self.stamp = data.stamp
            for i in self.graph.nodes():
                self.graph.nodes[i]['idleness'] += dev
                for j in self.graph.nodes[i]['future_visits'].keys():
                    self.graph.nodes[i]['future_visits'][j] -= dev

            for i, n in enumerate(data.node_id):
                self.graph.nodes[n]['idleness'] = 0.
                if data.robot_id[i] in self.graph.nodes[n]['future_visits'].keys():
                    if self.graph.nodes[n]['future_visits'][data.robot_id[i]] < 0.:
                        self.graph.nodes[n]['future_visits'].pop(data.robot_id[i])

            for i, r in enumerate(data.robot_id):
                if r not in self.robots.keys():
                    self.robots[r] = (data.node_id[i], s.stamp)
                elif self.robots[r][0] == data.node_id[i]:
                    self.robots[r] = (data.node_id[i], self.stamp)
                else:
                    last_delta = self.stamp - self.robots[r][1]
                    self.graph[self.robots[r][0]][data.node_id[i]]['duration'] = (1 - self.alpha_dur) * self.graph[self.robots[r][0]][data.node_id[i]]['duration'] + self.alpha_dur * last_delta
                    self.robots[r] = (data.node_id[i], self.stamp)

    def callback_next_task(self, req):
        node = req.node_done
        t = req.stamp
        bot = req.name
        prob = np.random.random_sample()
        print ('Time {}, Bot {}'.format(t, bot))
        if prob < s.eps_prob:
            succ = np.random.choice(list(s.graph.successors(node)))
            next_walk = [node, succ]
            next_departs = [t]
            print ('True Random')
        else:
            g_walk, _ = self.monte_carlo_sampling(node, self.plan_time)
            next_walk = [node]
            next_departs = []
            for i, n in enumerate(g_walk[1:]):
                if n == next_walk[-1]:
                    t += s.cf
                if n != next_walk[-1]:
                    next_walk.append(n)
                    next_departs.append(t)
                    t += np.ceil(self.graph[next_walk[-2]][next_walk[-1]]['duration'] / self.cf) * self.cf
            if len(g_walk) == 0:
                print ('Random')
                succ = np.random.choice(list(s.graph.successors(node)))
                next_walk = [node, succ]
                next_departs = [t]
            else:
                print ('Not random')
        for i, n in enumerate(next_walk[:-1]):
            self.graph.nodes[n]['future_visits'][bot] = next_departs[i] - req.stamp
        self.graph.nodes[next_walk[-1]]['future_visits'][bot] = self.plan_time
        return NextTaskBotResponse(next_departs, next_walk)

if __name__ == '__main__':
    rospy.init_node('tpbp_basic', anonymous = True)
    dirname = rospkg.RosPack().get_path('mrpp_sumo')
    done = False
    graph_name = rospy.get_param('/graph')
    g = nx.read_graphml(dirname + '/graph_ml/' + graph_name + '.graphml')

    priority_nodes = rospy.get_param('/priority_nodes').split(' ')
    time_periods = list(map(float, rospy.get_param('/time_periods').split(' ')))
    lambda_priority = float(rospy.get_param('/lambda_priority'))
    length_walk = float(rospy.get_param('/length_walk'))
    max_div = int(rospy.get_param('/max_divisions'))
    eps_prob = float(rospy.get_param('/eps_prob'))
    discount_factor = float(rospy.get_param('/discount_factor'))
    algo_name = rospy.get_param('/algo_name')
    s = TPBP_Basic(g, priority_nodes, time_periods, lambda_priority, length_walk, max_div, eps_prob, discount_factor, algo_name)

    rospy.Subscriber('at_node', AtNode, s.callback_idle)
    rospy.Timer(rospy.Duration(50), s.update_adj_matrix)
    rospy.Service('bot_next_task', NextTaskBot, s.callback_next_task)

    done = False
    while not done:
        done = rospy.get_param('/done')
        rospy.sleep(0.1)

