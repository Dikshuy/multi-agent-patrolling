#!/usr/bin/env python3

import rospy, rospkg, rosparam
import networkx as nx
import random as rn 
import sys, os
import time

from mrpp_sumo.srv import AddBot, RemoveBot
from mrpp_sumo.srv import AlgoReady

def add_bot_client(bot_name = 'bot1', node_name = '0'):
    rospy.wait_for_service('add_bot')
    try:
        add_bot_proxy = rospy.ServiceProxy('add_bot', AddBot)
        name = bot_name
        node = node_name
        bots_added = add_bot_proxy(name, node)
        return bots_added.done

    except rospy.ServiceException as e:
        print ("Service call failed: {}".format(e))

def remove_bot_client(bot_name = 'bot1', time = 0.):
    rospy.wait_for_service('remove_bot')
    try:
        remove_bot_proxy = rospy.ServiceProxy('remove_bot', RemoveBot)
        name = bot_name
        stamp = time
        bots_removed = remove_bot_proxy(name, stamp)
        return bots_removed.done

    except rospy.ServiceException as e:
        print ("Service call failed: {}".format(e))


if __name__ == '__main__':
    rospy.init_node('command_center', anonymous = True)
    graph_name = rospy.get_param('/graph')
    algo_name = str(rospy.get_param('/algo_name'))
    done = False
    dirname = rospkg.RosPack().get_path('mrpp_sumo')
    graph = nx.read_graphml(dirname + '/graph_ml/' + graph_name + '.graphml')
    sim_length = float(rospy.get_param('/sim_length'))
    
    #Waiting for Algo
    rospy.wait_for_service('algo_ready')
    is_ready = rospy.ServiceProxy('algo_ready', AlgoReady)
    algo_ready = False
    while not algo_ready:
        algo_ret = is_ready(algo_name)
        algo_ready = algo_ret.ready
        print ('Algorithm: {}, Ready {}'.format(algo_name, str(algo_ready)))
        time.sleep(2.)

    #Adding Bots
    init_bots = int(rospy.get_param('/init_bots'))
    bot_locations = rospy.get_param('/init_locations').split(' ')
    cur_bots = []
    for i in range(init_bots):
        if bot_locations == ['']:
            n = rn.choice(list(graph.nodes()))
        else:
            n = bot_locations[i]
        l = add_bot_client('bot_{}'.format(i), n)
        print (l)
        cur_bots.append('bot_{}'.format(i))
        time.sleep(1.0)


    while rospy.get_time() <= sim_length:
        rospy.sleep(0.01)

    rospy.set_param('done', True)