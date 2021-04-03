#!/usr/bin/env python3

'''
A ROS-wrapper to interact with sumo through traci from MRPP tasks
Initialization - 
1. Graph name
2. Number of robots  
'''

import os, sys
import rospy, rospkg
import networkx as nx
import traci
import configparser as CP
import random as rn
import time

from rosgraph_msgs.msg import Clock

from mrpp_sumo.srv import AddBot, AddBotResponse
from mrpp_sumo.srv import NextTaskBot, NextTaskBotResponse
from mrpp_sumo.srv import RemoveBot, RemoveBotResponse
from mrpp_sumo.msg import AtNode

class Sumo_Wrapper:

    def __init__(self, graph, output_file, output_file1):
        self.graph = graph
        self.edge_nodes = {}
        for e in self.graph.edges:
            self.edge_nodes[self.graph[e[0]][e[1]]['name']] = e
        self.stamp = 0.
        self.robots = []
        self.robot_add = []
        self.robot_rem = []
        self.routes = {}
        self.robots_stopped_already = []
        self.robots_just_stopped = []
        self.add_bot_service = rospy.Service('add_bot', AddBot, self.adding_bot)
        self.remove_bot_service = rospy.Service('remove_bot', RemoveBot, self.removing_bot)
        self.next_task_service = rospy.ServiceProxy('bot_next_task', NextTaskBot)
        self.output_file = open(output_file, 'a+')
        self.output_file1 = open(output_file1, 'a+')
        self.robots_in_traci = []

    #Adding bot at a specified location and then specifying its task
    def adding_bot(self, req):
        if not req.name in self.robots:
            self.robot_add.append((req.name, self.stamp))
            node = req.node
            if not node in self.graph.nodes:
                print ("Node not specified, assigning a node at random")
                node = rn.sample(list(self.graph.nodes), 1)[0]
            rospy.wait_for_service('bot_next_task')
            task_update = self.next_task_service(self.stamp, req.name, node)
            route = []
            departs = []
            for i in range(len(task_update.task) - 1):
                route.append(self.graph[task_update.task[i]][task_update.task[i + 1]]['name'])
                departs.append(float(task_update.departs[i]))
            self.robots.append(req.name)
            self.routes[req.name] = {'d': departs, 'r': route}
            return AddBotResponse(True)
        else:
            print ("Already a bot with the name - {}".format(req.name))
            return AddBotResponse(False)

    #Request for Next Task
    def next_task_update(self, vehicle, cur_edge):
        rospy.wait_for_service('bot_next_task')
        task_update = self.next_task_service(self.stamp, vehicle, self.edge_nodes[cur_edge][-1])
        route = []
        departs = []
        self.output_file.write(str(self.stamp) + "\n")
        self.output_file.write(str(vehicle) + "\n")
        self.output_file.write(' '.join(map(str, task_update.task)) + '\n')
        self.output_file.write(' '.join(map(str, task_update.departs)) + '\n')
        for i in range(len(task_update.task) - 1):
            route.append(self.graph[task_update.task[i]][task_update.task[i + 1]]['name'])
            departs.append(float(task_update.departs[i]))
        self.routes[vehicle] = {'d': departs, 'r': route}
        print(self.stamp, vehicle, self.routes[vehicle])

    def removing_bot(self, req):
        if req.name in self.robots:
            self.robot_rem.append((req.name, req.stamp))
            self.robots_in_traci.remove(req.name)
            return RemoveBotResponse(True)
        else:
            return RemoveBotResponse(False)


def main():
    rospy.init_node('sumo_wrapper', anonymous = True)
    dirname = rospkg.RosPack().get_path('mrpp_sumo')
    pub = rospy.Publisher('at_node', AtNode, queue_size = 10)
    pub_time = rospy.Publisher('/clock', Clock, queue_size = 10)
    done = False
    
    graph_name = rospy.get_param('/graph')
    file_name = rospy.get_param('/random_string')
    
    file_path = dirname + '/outputs/{}_command.in'.format(file_name)
    file_path1 = dirname + '/outputs/{}_visits.in'.format(file_name)
    g = nx.read_graphml(dirname + '/graph_ml/' + graph_name + '.graphml')
    s = Sumo_Wrapper(g, file_path, file_path1)

    one_way_temp = rospy.get_param('/one_way').split(' ')
    one_way_edges = {}
    for i in range(0, len(one_way_temp), 2):
        one_way_edges[g[one_way_temp[i]][one_way_temp[i + 1]]['name']] = {'edge': g[one_way_temp[i + 1]][one_way_temp[i]]['name'], 'blocked': False}
        one_way_edges[g[one_way_temp[i + 1]][one_way_temp[i]]['name']] = {'edge': g[one_way_temp[i]][one_way_temp[i + 1]]['name'], 'blocked': False}
    # sumo_startup = ['sumo', '-c', dirname + '/graph_sumo/{}.sumocfg'.format(graph_name), '--fcd-output', dirname + '/outputs/{}_vehicle.xml'.format(file_name)]
    sumo_startup = ['sumo-gui', '-c', dirname + '/graph_sumo/{}.sumocfg'.format(graph_name), '--fcd-output', dirname + '/outputs/{}_vehicle.xml'.format(file_name)]
    traci.start(sumo_startup)

    init_bots = int(rospy.get_param('/init_bots'))
    while len(s.robots) < init_bots:
        print ("{} robots are yet to be initialized".format(init_bots - len(s.robots)))

    time.sleep(1.0)
    while not done:
        done = rospy.get_param('done')
        s.stamp = traci.simulation.getTime()
        print (s.stamp)

        t = Clock()
        t.clock = rospy.Time(int(s.stamp))
        pub_time.publish(t)

        # in_sim_robots = traci.vehicle.getIDList()

        for _, w in enumerate(s.robot_rem):
            if w[1] <= s.stamp and w[0] in s.robots:
                s.robots.remove(w[0])
                if w[0] in s.robots_stopped_already:
                    s.robots_stopped_already.remove(w[0])
                traci.vehicle.remove(vehID = w[0])
                s.routes.pop(w[0])
        
        if len(s.robots) == 0:
            print ("No robots currently in the simulator. If you wish to terminate, set rosparam done to True")  

        else:
            stopped_bots = []
            added_bots = []
            for robot in s.robots:
                # print (robot, s.robots_in_traci, s.routes[robot], s.stamp)
                if (not robot in s.robots_in_traci) and (s.routes[robot]['d'][0] <= s.stamp):
                # if not robot in in_sim_robots:
                    if s.routes[robot]['r'][0] in one_way_edges.keys() and one_way_edges[s.routes[robot]['r'][0]]['blocked']:
                        pass

                    else:
                        added_bots.append(robot)
                        # print (s.routes[robot]['r'][:1])
                        traci.route.add(routeID = robot + '_edges', edges = s.routes[robot]['r'][:1])
                        traci.vehicle.add(vehID = robot, routeID = robot + '_edges')
                        traci.vehicle.setStop(vehID = robot, edgeID = s.routes[robot]['r'][0], pos = traci.lane.getLength(s.routes[robot]['r'][0] + '_0'), duration = 1000.)
                        s.routes[robot]['r'].pop(0)
                        s.routes[robot]['d'].pop(0)
                        s.robots_in_traci.append(robot)
                    # in_sim_robots = traci.vehicle.getIDList()
                elif traci.vehicle.isStopped(vehID = robot) and not robot in s.robots_just_stopped:
                    stopped_bots.append(robot)
                    s.robots_just_stopped.append(robot)

                elif traci.vehicle.isStopped(vehID = robot) and robot in s.robots_just_stopped and not robot in s.robots_stopped_already:
                    # print (traci.vehicle.getStops(vehID = robot))              
                    if len(s.routes[robot]['r']) == 0:
                        s.next_task_update(robot, traci.vehicle.getRoute(vehID = robot)[-1])
                    print (robot, s.routes[robot]['r'][0])
                    if s.routes[robot]['r'][0] in one_way_edges.keys() and one_way_edges[s.routes[robot]['r'][0]]['blocked']:
                        one_way_edges[one_way_edges[s.routes[robot]['r'][0]]['edge']]['blocked'] = True                            

                    else:
                        if s.routes[robot]['r'][0] in one_way_edges.keys():
                            one_way_edges[one_way_edges[s.routes[robot]['r'][0]]['edge']]['blocked'] = True                            
                        d = s.routes[robot]['d'].pop(0)
                        next_edges = [traci.vehicle.getRoute(vehID = robot)[-1], s.routes[robot]['r'].pop(0)]
                        traci.vehicle.setRoute(vehID = robot, edgeList = next_edges)
                        traci.vehicle.setStop(vehID = robot, edgeID = next_edges[0], pos = traci.lane.getLength(next_edges[0] + '_0'), duration = float(d - s.stamp))
                        traci.vehicle.setStop(vehID = robot, edgeID = next_edges[1], pos = traci.lane.getLength(next_edges[1] + '_0'), duration = 1000.)
                        s.robots_stopped_already.append(robot)
                        # stopped_bots.append(robot)
                
                elif not traci.vehicle.isStopped(vehID = robot) and robot in s.robots_stopped_already:
                    s.robots_stopped_already.remove(robot)
                    s.robots_just_stopped.remove(robot)

            
            if len(stopped_bots) + len(added_bots) > 0:
                msg = AtNode()
                msg.stamp = s.stamp
                msg.robot_id = stopped_bots
                msg.node_id = []
                for w in stopped_bots:
                    stop_lane = traci.vehicle.getLaneID(vehID = w)
                    stop_edge = stop_lane[:-2]
                    msg.node_id.append(s.edge_nodes[stop_edge][-1])

                msg.robot_id.extend(added_bots)
                for w in added_bots:
                    stop_edge = traci.vehicle.getRoute(vehID = w)[0]
                    msg.node_id.append(s.edge_nodes[stop_edge][0])
                pub.publish(msg)

                s.output_file1.write(str(s.stamp) + '\n')
                s.output_file1.write(' '.join(map(str, msg.node_id)) + '\n')
                s.output_file1.write(' '.join(map(str, msg.robot_id)) + '\n')

            for e in one_way_edges.keys():
                if traci.edge.getLastStepVehicleNumber(e) - traci.edge.getLastStepHaltingNumber(e) > 0:
                    one_way_edges[one_way_edges[e]['edge']]['blocked'] = True
                    # traci.edge.setParameter(one_way_edges[e]['edge'], 'color', 'blue')
                else:
                    one_way_edges[one_way_edges[e]['edge']]['blocked'] = False


            traci.simulationStep()
    s.output_file.close()
    s.output_file1.close()
    traci.close()

if __name__== '__main__':
    main()