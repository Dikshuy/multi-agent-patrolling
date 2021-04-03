#!/usr/bin/env python3


'''
NOT NECESSARY
'''
import rospy
import rospkg
import os, sys
from mrpp_sumo.msg import AtNode


class Record_Data:

    def __init__(self):
        rospy.init_node('record_data', anonymous = True)
        self.dirname = rospkg.RosPack().get_path('mrpp_sumo')
        self.file_name = rospy.get_param('/random_string')
        self.dest_file = self.dirname + '/outputs/{}_visits.in'.format(self.file_name)
        rospy.Subscriber('at_node', AtNode, self.callback)
        self.stamp = -1.
        self.f = open(self.dest_file, 'a+')

    def callback(self, data):
        if self.stamp < data.stamp:
            self.stamp = data.stamp
            self.f.write(str(data.stamp) + '\n')
            self.f.write(' '.join(map(str, data.node_id)) + '\n')
            self.f.write(' '.join(map(str, data.robot_id)) + '\n')

if __name__ == '__main__':
    t = Record_Data()
    done = False
    while not done:
        done = rospy.get_param('/done')
        
    t.f.close()