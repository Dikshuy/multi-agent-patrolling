import pandas as pd
import matplotlib.pyplot as plt
import xlsxwriter
import numpy as np
import os
import rospkg


pkgdir = rospkg.RosPack().get_path('mrpp_sumo')
algo_name = "with_intent"
config_folder = pkgdir + "/config/" + algo_name + "/"
data_folder = pkgdir + "/post_process/" + algo_name + "/"

graph_name = ['complex_final']
init_bots = [1, 3, 5, 7]
no_of_deads = [1,12,25]
no_runs = 2




for dead in no_of_deads:
	for  agent in init_bots:
		for run in range(no_runs):
			filename =  algo_name+ str(len(no_of_deads)*no_runs*init_bots.index(agent)+no_of_deads.index(dead)*2+run+1)
			# print("for ",dead, "dead ",agent, "agents ",run,"run_id 	",algo_name,len(no_of_deads)*no_runs*init_bots.index(agent)+no_of_deads.index(dead)*2+run+1)
			print(np.load(data_folder+filename+"/data.npy"))



		

