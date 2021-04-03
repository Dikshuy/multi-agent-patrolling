#!/usr/bin/env python

'''
Online module of TPBP Algo on walks
'''
import rospy
import rospkg
import pdb
import os
import networkx as nx
#import ConfigParser as CP
from patrol_messages.msg import *
from patrol_messages.srv import *
from sets import Set
import numpy
import time
from collections import OrderedDict

robot_to_node = OrderedDict()


def max_nodes_idle_list(nodList, nIdleList, iterList):
	if iterList >= 0:
		ind = numpy.argpartition(nIdleList, -iterList)[-iterList:]
		node_list = [nodList[i][0] for i in ind]
		return node_list

def find_max_2d(nList):
	max_val = 0
	max_val_ind = -1
	for r in range(len(nList)):
		if nList[r][1] > max_val:
			max_val=nList[r][1]
			max_val_ind = r
	return max_val_ind

def get_last_node(data_dict, comp_key):
	for key, value in data_dict.items():
		if key == comp_key:
			return value[1]


def tpbp_reward(g, algo_params, walk, priority_nodes, time_periods, assigned):
    nodes = list(Set(walk))
    term1 = 0.
    term2 = 0.
    for i in nodes:
        term1 += g.node[i]['idleness']
        if i in priority_nodes:
            j = priority_nodes.index(i)
            if not assigned[j]:
                term2 += max(g.node[i]['idleness'] - time_periods[j], 0)
    term3 = 0.
    for i in range(len(priority_nodes)):
        if not assigned[i] and not priority_nodes[i] in nodes:
            dist = numpy.inf
            for j in nodes:
                temp = nx.shortest_path_length(g, j, priority_nodes[i], 'length')
                if temp < dist:
                    dist = temp
            term3 += dist
    term4 = 0.
    for i in range(len(walk) - 1):
        term4 += g[walk[i]][walk[i + 1]]['length']
    return numpy.dot([term1, term2, term3, term4], algo_params)

def tpbp_walk(road_graph, algo_params, cur_node, priority_nodes, time_periods, assigned, folder_path, rob_id, num_robs):
    best_reward = -numpy.inf
    best_walk = []
    rospy.loginfo(priority_nodes)
    rospy.loginfo(assigned)
    # print 'Not Here', rob_id, cur_node, priority_nodes, assigned
    for j in range(len(priority_nodes)):
        # print 'test', j, len(priority_nodes)
        if not assigned[j]:
            # print 'Here', rob_id
            # if cur_node in priority_nodes:
                # print 'Here1'
            valid_trails = '/valid_trails_{}_{}_{}.in'.format(cur_node, priority_nodes[j], str(int(time_periods[priority_nodes.index(cur_node)])))
            #print 'file name: '+folder_path + valid_trails
            # try:
            with open(folder_path + valid_trails, 'r') as f:
                # print 'Here2', valid_trails

                count = 0
                for line in f:
                    #print line
                    count += 1
                    line1 = line.split('\n')
                    line2 = line1[0].split(' ')
                    r = tpbp_reward(road_graph, algo_params, line2, priority_nodes, time_periods, assigned)
                    if r > best_reward:
                        best_reward = r
                        best_walk = line2
                            # print best_reward, best_walk
                            # rospy.loginfo(' best walk set to %s', best_walk)
                # except Exception, e:
                #     rospy.logerr(" Exception in onpening valid trail file %s",valid_trails)
                #     rospy.logerr(e)
            
            # else:
			# 	#priority node with max idlesness value
			# 	idleness_pn = [road_graph.node[r]['idleness'] for r in priority_nodes]
			# 	index_idleness = numpy.argmax(idleness_pn)
			# 	#find shortest_path between priority_nodes and curr node
			# 	best_walk = nx.shortest_path(road_graph, cur_node, priority_nodes[index_idleness], weight = 'length')
            # print 'To Here'
        else:
            print str(j)+' assigned '
        print 'to test'

    # If all nodes are assigned
	if all([x == True for x in assigned]):
		idleness_value_nn = []
		for nod in road_graph.nodes():
			if nod not in priority_nodes:
				idleness_value_nn.append([nod, road_graph.node[nod]['idleness']])
		norm_node_array = max_nodes_idle_list(idleness_value_nn, [i[1] for i in idleness_value_nn], (num_robs-len(priority_nodes)))

		if rob_id not in robot_to_node: #check and handle here
			min_dist_val = 1000000
			short_path_node = ''
			for node_id_max in norm_node_array:
				dist_val = nx.shortest_path_length(road_graph, cur_node, node_id_max,  weight = 'length')
				if dist_val < min_dist_val:
					max_dist_val = dist_val
					short_path_node = node_id_max
		       	best_walk = nx.shortest_path(road_graph, cur_node, short_path_node,  weight = 'length')
			road_graph.node[short_path_node]['idleness'] = 0.0
		else:
			curr_node_val = get_last_node(robot_to_node, rob_id)
			min_dist_val = 1000000
			short_path_node = ''
			for node_id_max in norm_node_array:
				dist_val = nx.shortest_path_length(road_graph, cur_node, node_id_max,  weight = 'length')
				if dist_val < min_dist_val:
					max_dist_val = dist_val
					short_path_node = node_id_max
			best_walk = nx.shortest_path(road_graph, curr_node_val, short_path_node,  weight = 'length')
			road_graph.node[short_path_node]['idleness'] = 0.0

    print 'Final Selected best walk is ' + str( best_walk), rob_id, j
    robot_to_node[rob_id] = [best_walk[0], best_walk[-1]]
    return best_walk

class TPBP:
    def __init__(self):
        self.is_initial_params_received = False
        self.initial_parm_subs = rospy.Subscriber('pmm_to_algo_params', Initialize_patrol, self.initializaiton_params_CB)
        self.timer = rospy.Timer(rospy.Duration(1),self.timed_algo_setup_cb, oneshot=True)

        rospy.sleep(1.0)
        #self.add_remove_robot_subs = rospy.Subscriber('team_update_topic_name', team_update, self.team_update_CB)
        self.team_update_to_tpbp_service = rospy.Service('team_update_to_tpbp', TeamUpdateRobotSrv, self.team_update_CB)

        
        rospy.loginfo("TPBP Constructor done!!!")

    def timed_algo_setup_cb(self, timer):
        rospy.logdebug_throttle(1,"timer callback")
        self.algorithm_setup()

    def team_update_CB(self, request):
		rospy.wait_for_service('team_update_to_sumo')
		self.team_update_to_sumo_proxy = rospy.ServiceProxy('team_update_to_sumo', TeamUpdateRobotSrv)
		try:
			self.team_update_to_sumo_proxy(request.robot_id_list, request.robot_id, request.start_node_id, request.rem_check)
			#print(self.team_update_to_sumo_proxy(request.robot_id_list, request.robot_id, request.start_node_id))
		except Exception as e:
			rospy.logerr(e)

		if(len(self.robot_ids) < len(request.robot_id_list)):           #if a robot is added
			indx = 0
			self.robot_ids = request.robot_id_list
			#rospy.loginfo("Computing initial tpbp_walk for robot %s", request.robot_id)
			if request.start_node_id in self.graph.node:
				self.robot_cur_walks[request.robot_id] = tpbp_walk(self.graph, self.algo_params, request.start_node_id, self.priority_nodes_cur, self.time_periods, self.assigned, self.dest_folder, request.robot_id, len(self.robot_ids))
			else:
				for i in self.assigned:
					if self.assigned[i] == False:
						indx = i
				self.robot_cur_walks[request.robot_id] = tpbp_walk(self.graph, self.algo_params, self.priority_nodes_cur[indx], self.priority_nodes_cur, self.time_periods, self.assigned, self.dest_folder, request.robot_id, len(self.robot_ids))

			#rospy.loginfo('PRINTING Dict')
			#rospy.loginfo(self.robot_cur_walks)
			if self.robot_cur_walks[request.robot_id][-1] in self.priority_nodes:
				self.assigned[self.priority_nodes.index(self.robot_cur_walks[request.robot_id][-1])] = True
			self.task_list[request.robot_id] = map(str, self.robot_cur_walks[request.robot_id])
			#rospy.loginfo(self.task_list)
			#print("SERVICE MESSAGE")
			#print(self.tpbp_to_pmm_proxy(map(str, self.robot_cur_walks[request.robot_id]), request.robot_id))
			self.tpbp_to_pmm_proxy(map(str, self.robot_cur_walks[request.robot_id]), request.robot_id)

		elif(len(self.robot_ids) > len(request.robot_id_list)):          #if a robot is removed
			self.robot_ids = request.robot_id_list
			print("Robot ids after removal: "+ str(self.robot_ids), request.rem_check)
			if self.robot_cur_walks[request.robot_id][-1] in self.priority_nodes:
				self.assigned[self.priority_nodes.index(self.robot_cur_walks[request.robot_id][-1])] = False
			del self.robot_cur_walks[request.robot_id]
			del self.task_list[request.robot_id]
			del robot_to_node[request.robot_id]



		return TeamUpdateRobotSrvResponse("Team Update to TPBP received")


    #def algorithm_setup(self, dirname, folder, num_vehicles, priority_nodes, graph_name, algo, algo_params, time_periods):
    def algorithm_setup(self):
        rospy.sleep(2)
        #rate = rospy.Rate(10) # 10hz
        rospy.loginfo("Algorithm setup initiated: TPBP ")
        rospy.loginfo("%s, %s, %s, %s, %s, %s, %s, %s",self.dirname, self.folder, self.num_robots, self.priority_nodes, self.graph_name, self.algo, self.algo_params, self.time_periods)
        # self.algo = algo
        print("Inside algo_setup Robot ids: "+ str(self.robot_ids))
        if self.algo != "tpbp_walk":
            rospy.logwarn(" Algorithm is not TPBP_WALK. Received algo is <%s>", self.algo)
            return
        # self.dirname = dirname
        # self.folder = folder
        self.dest_folder = self.dirname + '/processing/' + self.folder
        # self.num_robots = num_vehicles
        # self.priority_nodes = priority_nodes
        # self.time_periods = time_periods
        self.graph = nx.read_graphml(self.dirname + '/graph_ml/' + self.graph_name + '.graphml')
        for node in self.graph.nodes():
            self.graph.node[node]['idleness'] = 0.
            rospy.loginfo_throttle(1, ' node idleness set to 0')
        # self.algo_params = algo_params
        self.stamp = 0.0


	#Choosing dummy nodes for randomization
        self.num_dummy_nodes = 1 #Number of dummy nodes
        self.reshuffle_time = 1800. #Expected Time between reshuffles
        self.dummy_time_period = [self.time_periods[0]] * self.num_dummy_nodes
        self.time_periods.extend(self.dummy_time_period)        
        self.nodes = list(self.graph.nodes())
        self.non_priority_nodes = [item for item in self.nodes if item not in self.priority_nodes]


        self.dummy_nodes = numpy.random.choice(self.non_priority_nodes, self.num_dummy_nodes)
        self.priority_nodes_cur = self.priority_nodes[:]
        self.priority_nodes_cur.extend(self.dummy_nodes)
        self.priority_nodes_prev = self.priority_nodes_cur[:]
        self.reshuffle_next = numpy.random.poisson(self.reshuffle_time)

        self.assigned = []
        for i in self.priority_nodes_cur:
            self.assigned.append(False)
        self.robot_cur_walks = {}
        #self.list_robot_pub = {}
        self.task_list = {}
        
        #self.task_done_subs = rospy.Subscriber('/task_done', TaskDone, self.task_done_CB)
        self.tpbp_service = rospy.Service('pmm_to_tpbp', TaskDoneSrv, self.task_done_CB)
        rospy.logdebug(" Constructing TPBP _line_=91 ")       
        for index, ith_robot in zip(range(len(self.robot_ids)), self.robot_ids):
            rospy.loginfo("Computing initial tpbp_walk for robot %s", ith_robot)
            time_start = time.clock()
            self.robot_cur_walks[ith_robot] = tpbp_walk(self.graph, self.algo_params, self.priority_nodes_cur[index], self.priority_nodes_cur, self.time_periods, self.assigned, self.dest_folder, ith_robot, len(self.robot_ids))
            print(self.robot_cur_walks)
            rospy.loginfo('PRINTING Dict')
            rospy.loginfo(self.robot_cur_walks)
            if self.robot_cur_walks[itr_robot][-1] in self.priority_nodes:
                self.assigned[self.priority_nodes.index(self.robot_cur_walks[itr_robot_id][-1])] = True
            time_end = time.clock()
            time_spend = time_end - time_start
            rospy.loginfo(" Completed computation of tpbp_walk. Time spent %f", time_spend)                        
            #self.task_list.append(map(int, self.robot_cur_walks[i]))
            rospy.loginfo("TASK_LIST PRINT:")
            self.task_list[ith_robot] = map(str, self.robot_cur_walks[ith_robot])
            rospy.loginfo(self.task_list) 
        rospy.loginfo(self.robot_cur_walks) 
        rospy.loginfo("Line _163_ Robot ids: %s", str(self.robot_ids))
        for ith_robot in self.robot_ids:
            rospy.loginfo("inside for of robot id %s", ith_robot)
            #topic_name = '/robot_{}/next_task_to_pmm'.format(ith_robot)
            #rospy.loginfo("creating the publisher for robot %s on topic %s", ith_robot, topic_name)
            try:
                #robot_i_next_task_publisher = rospy.Publisher('/robot_{}/next_task_to_pmm'.format(ith_robot), NextTask, latch=True, queue_size = 10)            
                #print(robot_i_next_task_publisher)
                # next_task_msg = NextTask()
                # next_task_msg.task = self.task_list[ith_robot] 
                # next_task_msg.robot_id = ith_robot           
                # rospy.loginfo(self.task_list[ith_robot])

                rospy.wait_for_service('tpbp_to_pmm')
                self.tpbp_to_pmm_proxy = rospy.ServiceProxy('tpbp_to_pmm', NextTaskSrv)
                try:
                    print(self.tpbp_to_pmm_proxy(self.task_list[ith_robot], ith_robot))
                    # robot_i_next_task_publisher.publish(next_task_msg)
                    #rate.sleep()
                    rospy.loginfo(self.task_list[ith_robot])
                    rospy.loginfo(ith_robot)
                except Exception,e:
                    rospy.logwarn(" Not able to publish the next task for robot %s Exception is:", ith_robot)
                    rospy.logerr(e)            
                #self.list_robot_pub[ith_robot]=robot_i_next_task_publisher
            except Exception,e:
                rospy.logerr(e)
                rospy.logerr("unable to create service client for robot %s for topic %s", ith_robot)

        rospy.loginfo("TPBP Setup done!!!")

    def __del__(self):
        rospy.logwarn(" TPBP destructor called ")
        print("TPBP destructor called")
    
    def task_done_CB(self, request):
        rospy.loginfo_throttle(1,"ENTERED /task_done CALLBACK. ")

        if self.stamp <= request.stamp:# to ignore out-of-order messages from the simulator
            #Update Idleness
            dev = request.stamp - self.stamp
            self.stamp = request.stamp
            for i in self.graph.nodes():
                self.graph.node[i]['idleness'] += dev
            for i in range(len(request.node_id)):
                    self.graph.node[str(request.node_id[i])]['idleness'] = 0
            
            #Randomization Code
            if self.stamp >= self.reshuffle_next:
                self.reshuffle_next = self.stamp + numpy.random.poisson(self.reshuffle_time)
                self.dummy_nodes = numpy.random.choice(self.non_priority_nodes, self.num_dummy_nodes)
                self.priority_nodes_prev = self.priority_nodes_cur[:]
                self.priority_nodes_cur = self.priority_nodes[:]
                self.priority_nodes_cur.extend(self.dummy_nodes)
                self.assigned_prev = self.assigned[:]

                for i, n in enumerate(self.dummy_nodes):
                    self.assigned[len(self.priority_nodes) + i] = False
                    if n in self.priority_nodes_prev:
                        n_prev = self.priority_nodes_prev.index(n)
                        self.assigned[len(self.priority_nodes) + i] = self.assigned_prev[n_prev]


            #Update Walk If Necessary            
            for i, itr_robot_id in zip(range(len(request.robot_id)),request.robot_id):
                print_msg = "Checking for updates to walk of robot " + itr_robot_id
                print(print_msg)                
                n = request.node_id[i]
                m = self.robot_cur_walks[request.robot_id[i]][0]
                print str(n), m
                if str(n) == m:  
                    print("robot_cur_walks b4 " , self.robot_cur_walks[request.robot_id[i]])                   
                    self.robot_cur_walks[request.robot_id[i]].pop(0)    
                    print("robot_cur_walks after " , self.robot_cur_walks[request.robot_id[i]])        

                if len(self.robot_cur_walks[request.robot_id[i]]) == 0:
					if str(n) in self.priority_nodes:
						self.assigned[self.priority_nodes.index(str(n))] = False
					self.robot_cur_walks[itr_robot_id] = tpbp_walk(self.graph, self.algo_params, str(n), self.priority_nodes_cur, self.time_periods, self.assigned, self.dest_folder, itr_robot_id, len(self.robot_ids))
					#rospy.loginfo('PRINTING Dict')
					#rospy.loginfo(self.robot_cur_walks)
					if self.robot_cur_walks[itr_robot_id][-1] in self.priority_nodes:
						self.assigned[self.priority_nodes.index(self.robot_cur_walks[itr_robot_id][-1])] = True
					#print("SERVICE MESSAGE")
					self.tpbp_to_pmm_proxy(map(str, self.robot_cur_walks[itr_robot_id]), itr_robot_id)
					#print(self.tpbp_to_pmm_proxy(map(str, self.robot_cur_walks[itr_robot_id]), itr_robot_id))
    
        else:
            rospy.logwarn_throttle(1, ' self.stamp is NOT less than request.stamp !')
        return TaskDoneSrvResponse("Task Done Params Received by TPBP!!!")
    
    def initializaiton_params_CB(self, init_msg):
        self.dirname = rospkg.RosPack().get_path('patrol_algo')
        self.folder = init_msg.folder
        self.num_robots = init_msg.num_robots
        self.priority_nodes = init_msg.priority_nodes.split(' ')
        self.graph_name = init_msg.graph
        self.algo = init_msg.algo
        self.algo_params = map(float, init_msg.algo_params.split(' '))
        self.time_periods = map(float, init_msg.time_periods.split(' '))
        self.robot_ids = init_msg.robot_ids.split(' ')
        
        #self.reshuffle_time = float(init_msg.reshuffle_time)
        #self.num_dummy_nodes = int(init_msg.num_dummy_nodes)
        #self.initial_parm_subs.unregister()
        rospy.loginfo("Received Initial params: %s, %s, %s, %s, %s, %s, %s, %s, %s", self.dirname, self.folder, self.num_robots, self.priority_nodes, self.graph_name, self.algo, self.algo_params, self.time_periods, self.robot_ids)    

def valid_walks_done_CB(msg):
    if msg.valid_walks_done:
        print("TPBP_ONLINE Received Start confirmation")
        t = TPBP()   
        done = False
        del t
    
if __name__ == '__main__':
    rospy.init_node('tpbp_online', anonymous = False)   
    valid_walks_done_tpbp = rospy.Subscriber('valid_walks_topic', ValidWalksDone, valid_walks_done_CB)
    rospy.spin()

    
