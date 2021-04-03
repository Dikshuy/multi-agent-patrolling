#!/usr/bin/env python

import tpbp_offline as tp
#import ConfigParser as CP
import networkx as nx
import rospkg
import rospy
import sys

from patrol_messages.msg import *

def callback(msg):
    print("START")
    pub = rospy.Publisher('valid_walks_topic', ValidWalksDone, queue_size=10)
    dirname = rospkg.RosPack().get_path('patrol_algo')
    print msg.folder
    dest_folder = dirname + '/processing/' + msg.folder
    priority_nodes = msg.priority_nodes.split(' ')
    time_periods = map(float, msg.time_periods.split(' '))
    graph = nx.read_graphml(dirname + '/graph_ml/' + msg.graph + '.graphml')
    tp.all_valid_trails(graph, priority_nodes, time_periods, dest_folder)

    msg1 = ValidWalksDone()
    msg1.valid_walks_done = True
    pub.publish(msg1)
    print("END")
    sub.unregister()


rospy.init_node('algo_module')
sub = rospy.Subscriber('pmm_to_algo_params', Initialize_patrol, callback)
rospy.spin()


'''
def main(sim_set):


    rospy.init_node('algo_module')

    sub = rospy.Subscriber('<To be decided>', Inititalize_patrol, callback)

    dirname = rospkg.RosPack().get_path('mrpp_algos')
    config_file = dirname + '/config.txt'
    config = CP.ConfigParser()
    config.read(config_file)
    params = {}
    for option in config.options(sim_set):
        params[option] = config.get(sim_set, option)
    if params['algo'] != 'tpbp_walk':
        return
    dest_folder = dirname + '/' + params['folder']
    priority_nodes = params['priority_nodes'].split(' ')
    time_periods = map(float, params['time_periods'].split(' '))
    graph = nx.read_graphml(dirname + '/graph_ml/' + params['graph'] + '.graphml')
    tp.all_valid_trails(graph, priority_nodes, time_periods, dest_folder)
if __name__ == '__main__':
    if len(sys.argv[1:]) >= 1:
        main(sys.argv[1])
    else:
        print 'Please pass the appropriate arguments'

'''
