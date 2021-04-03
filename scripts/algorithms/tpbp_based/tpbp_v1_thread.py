#!/usr/bin/env python3


'''
Infinite Horizon Cost

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

import threading

#Incomplete
class MCTS(threading.Thread):
    def __init__(self, thread_id, algo_class, adj_temp, orig, div_cur, vals_n, future_visit_final, dest):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.algo_class = algo_class
        self.adj_temp = adj_temp
        self.vals_n = vals_n[:]
        self.future_visit_final = future_visit_final[:]
        self.dest = dest
        self.div_cur = div_cur
        self.orig_id = self.algo_class.nodes.index(orig)        

    def run(self):

        g_n = np.zeros([self.algo_class.N, self.div_cur + 1])

        walks = []
        for i in range(self.algo_class.N):
            walks.append([])
            for j in range(self.div_cur + 1):
                walks[i].append([])
                 
        for i in range(self.div_cur + 1):
            for j in range(self.algo_class.N):
                g_e = -1. * np.ones(self.algo_class.N)
                for k in range(self.algo_class.N):
                    if self.adj_temp[k, j, i] >= 0:
                        g_e[k] = g_n[k, self.adj_temp[k, j, i]] + self.vals_n[k, self.adj_temp[k, j, i], j, i]
                        if i > 0 and len(walks[k][self.adj_temp[k, j, i]]) == 0:
                            g_e[k] = -1.
                max_g = np.max(g_e)
                if max_g >= 0.:
                    args_temp = np.argwhere(g_e == max_g).flatten().tolist()
                    # print(i, j, args_temp)
                    #Random Selection    
                    sel_k = np.random.choice(args_temp)
                    vals_l = np.copy(self.vals_n[sel_k, self.adj_temp[sel_k, j, i]])

                    #Node Weight Update 
                    if i <= self.div_cur and i * self.algo_class.cf > self.future_visit_final[j]:
                        for n in range(i, self.div_cur + 1):
                            vals_l[j, n] = float(self.div_cur - n)/self.div_cur * (n - i) * self.algo_class.cf 
                            if self.algo_class.nodes[j] in self.algo_class.priority_nodes:
                                    vals_l[j, n] += float(self.div_cur - n)/self.div_cur * self.algo_class.lambda_priority * min((n - i) * self.algo_class.cf, self.algo_class.time_periods[self.algo_class.priority_nodes.index(self.algo_class.nodes[j])])
                    
                    elif i <= self.div_cur and i * self.algo_class.cf < self.future_visit_final[j]:
                        fut = self.future_visit_final[j]
                        nod = self.algo_class.nodes[j]
                        for val in list(self.algo_class.graph.nodes[nod]['future_visits'].values()):
                            if val >  i * self.algo_class.cf and val < fut:
                                fut = val
                        for n in range(i, int(min(np.floor(fut / self.algo_class.cf), self.div_cur + 1))):
                            vals_l[j, n] = float(fut - n * self.algo_class.cf)/(self.algo_class.cf * self.div_cur) * (n - i) * self.algo_class.cf
                            if self.algo_class.nodes[j] in self.algo_class.priority_nodes:
                                vals_l[j, n] += float(fut - n * self.algo_class.cf)/(self.algo_class.cf * self.div_cur) * self.algo_class.lambda_priority * min((n - i) * self.algo_class.cf, self.algo_class.time_periods[self.algo_class.priority_nodes.index(self.algo_class.nodes[j])])


                    self.vals_n[j, i] = vals_l
                    g_n[j, i] = max_g
                    walks[j][i] = walks[sel_k][self.adj_temp[sel_k, j, i]][:]
                    walks[j][i].append(j)


        if self.dest == []:
            self.dest = self.algo_class.nodes
        dest_ids = []
        for n in self.dest:
            dest_ids.append(self.algo_class.nodes.index(n))
        max_g = np.max(g_n[dest_ids, self.div_cur])
        args_temp = np.argwhere(g_n[dest_ids, self.div_cur] == max_g).flatten().tolist()

        #Random Selection
        sel_dest = dest_ids[np.random.choice(args_temp)]
        walk_id = walks[sel_dest][self.div_cur]
        walk = []
        for i in walk_id:
            walk.append(self.algo_class.nodes[i])
        self.walk, self.max_g = walk, max_g


class TPBP_Basic:
    def __init__(self, g, priority_nodes, time_periods, lambda_priority, length_walk, max_div, eps_prob, discount_factor, algo_name, file_path):

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
        self.file_path = file_path
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
                        tdiv = int(np.ceil(t / self.cf))
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
                    if m * self.cf >= future_visit_final[i]:
                        vals_n[orig_id, 0, i, m] = float(div_cur - m)/div_cur * (m * self.cf - future_visit_final[i])
                        if w in self.priority_nodes:
                            vals_n[orig_id, 0, i, m] += float(div_cur - m)/div_cur * self.lambda_priority * min(m * self.cf - future_visit_final[i], self.time_periods[self.priority_nodes.index(w)])
                    
                    else:
                        fut = future_visit_final[i]
                        bef = 0
                        for val in self.graph.nodes[w]['future_visits'].values():
                            if val > m * self.cf and val < fut:
                                fut = val
                            elif val <= m * self.cf and val > bef:
                                bef = val

                        vals_n[orig_id, 0, i, m] = float(fut - m * self.cf)/len_cur * (m * self.cf - bef)
                        if w in self.priority_nodes:
                            vals_n[orig_id, 0, i, m] += float(fut - m * self.cf)/len_cur * self.lambda_priority * min(m * self.cf - bef, self.time_periods[self.priority_nodes.index(w)])

        if min(future_visit_final) > len_cur:
            return g_walk, g_max

        threads = []
        for i in range(self.num_samples):
            threads.append(MCTS(i, self, adj_temp, orig, div_cur, vals_n[:], future_visit_final[:], dest))
        for i in range(self.num_samples):
            threads[i].start()
        for i in range(self.num_samples):
            threads[i].join()
        walks = []
        gs = []

        for i in range(self.num_samples):
            walks.append(threads[i].walk)
            gs.append(threads[i].max_g)
        
        g_id = np.argmax(gs)
        g_walk = walks[g_id]
        g_max = gs[g_id]
        return g_walk, g_max

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
        print ('Time {}, Bot {}, Node {}'.format(t, bot, node))
        if prob < s.eps_prob:
            succ = np.random.choice(list(s.graph.successors(node)))
            next_walk = [node, succ]
            next_departs = [t]
            # print ('True Random')
        else:
            g_walk, _ = self.monte_carlo_sampling(node, self.plan_time)
            print (g_walk)
            next_walk = [node]
            next_departs = []
            for i, n in enumerate(g_walk[1:]):
                if n == next_walk[-1]:
                    t += s.cf
                if n != next_walk[-1]:
                    next_walk.append(n)
                    next_departs.append(t)
                    # print (n)
                    # print (self.graph[next_walk[-2]][next_walk[-1]]['duration'])
                    t += np.ceil(self.graph[next_walk[-2]][next_walk[-1]]['duration'] / self.cf) * self.cf
            if len(next_walk) == 1:
                # print ('Random')
                succ = np.random.choice(list(s.graph.successors(node)))
                next_walk = [node, succ]
                next_departs = [t]
            # else:
                # print ('Not random')
        for i, n in enumerate(next_walk[:-1]):
            self.graph.nodes[n]['future_visits'][bot] = next_departs[i] - req.stamp
        self.graph.nodes[next_walk[-1]]['future_visits'][bot] = self.plan_time
        return NextTaskBotResponse(next_departs, next_walk)

if __name__ == '__main__':
    rospy.init_node('tpbp_v1_thread', anonymous = True)
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
    file_name = rospy.get_param('/random_string')
    file_path = dirname + '/outputs/{}_command.in'.format(file_name)
    s = TPBP_Basic(g, priority_nodes, time_periods, lambda_priority, length_walk, max_div, eps_prob, discount_factor, algo_name, file_path)

    rospy.Subscriber('at_node', AtNode, s.callback_idle)
    rospy.Timer(rospy.Duration(50), s.update_adj_matrix)
    rospy.Service('bot_next_task', NextTaskBot, s.callback_next_task)

    done = False
    while not done:
        done = rospy.get_param('/done')
        # rospy.sleep(0.1)