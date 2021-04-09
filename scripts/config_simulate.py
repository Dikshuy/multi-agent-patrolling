#!/bin/usr/env python3

import configparser as CP
import os
import rospkg
import glob

dir_name = rospkg.RosPack().get_path('mrpp_sumo')
config_files = glob.glob(dir_name + '/config/without_intent_cr/without_intent_cr*.yaml')
count = 0
for conf in config_files:
    os.system('xterm -e "{}/tpbp.sh" {}'.format(dir_name, conf))
    count += 1
    print ('{} Done {}'.format(count, conf))
