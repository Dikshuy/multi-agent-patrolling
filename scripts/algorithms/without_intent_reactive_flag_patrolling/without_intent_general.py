#!/usr/bin/env python3

'''
Without intent
'''

import os
import time
import rospy
import rospkg
import ruamel.yaml
yaml = ruamel.yaml.YAML()

rospack = rospkg.RosPack()
agents = [1,2,3,4,5,6]
no_fails = [0,1,3,5,8,12]
yaml_filename = "without_intent_test.yaml"
data_path = "/scripts/algorithms/without_intent_patrolling/data"
os.system("rm -rf "+rospack.get_path('mrpp_sumo')+data_path)
os.system("mkdir "+rospack.get_path('mrpp_sumo')+data_path)

for agent in agents:
	os.system("rm -rf "+rospack.get_path('mrpp_sumo')+data_path+"/"+str(agent)+"agents")
	os.system("mkdir "+rospack.get_path('mrpp_sumo')+data_path+"/"+str(agent)+"agents")
	with open(rospack.get_path('mrpp_sumo')+"/config/"+yaml_filename) as f:
		data = yaml.load(f)
		data["init_bots"] = agent
	with open(rospack.get_path('mrpp_sumo')+"/config/"+yaml_filename, "w") as f:
		yaml.dump(data, f)

	for dead in no_fails:
		os.system("rm -rf "+rospack.get_path('mrpp_sumo')+data_path+"/"+str(agent)+"agents/"+str(dead)+"deads")
		os.system("mkdir "+rospack.get_path('mrpp_sumo')+data_path+"/"+str(agent)+"agents/"+str(dead)+"deads")
		os.system("rosparam load "+rospack.get_path('mrpp_sumo')+"/config/"+yaml_filename)
		os.system("rosrun mrpp_sumo without_intent_data_collection.py "+str(dead) + " "+rospack.get_path('mrpp_sumo')+data_path+"/"+str(agent)+"agents/"+str(dead)+"deads")