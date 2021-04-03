# Additional Readme to run one way scenario with tpbp algorithm

- Kindly note that the entire tpbp algorithm is combined into 'tpbp_final.py' script in ./scripts/algorithms/tpbp folder 
- To run the simulation follow the following steps 
    - create a config file [file_name].yaml containing simulation parameters in the form of 'test_one_way.yaml' available in ./config folder and place it in the ./config folder

    - execute the following in the terminal
      
            ./tpbp_one_way_tpbp.sh ./config/[file_name].yaml


## ROS Parameters required for simulation

- graph: you can add any graph (just the name of the file without extension) from ./graph_ml folder
- init_bots: number of agents initially in the simulation
- init_locations: starting node ids for the agents in the form of a string with ' ' spacing in between (ex: '0 5 10' for init_bots equal to 3), alternatively it can be set to '' in which case the starting positions are picked randomly
- use_sim_time: true 
- done: false
- sim_length: length of the simulation in seconds
- random_string: name of the output files and folders generated during the simulation
- algo_name: tpbp
- priority_nodes: node ids of priority nodes with ' ' spacing in between
- time_periods: time period of the priority nodes with ' ' spacing in between
- coefficients: values of c1, c2, c3 and c4 with ' ' spacing in between
- num_dummy_nodes: number of dummy priority nodes considered for randomisation
- reshuffle_time: time between reshuffling of dummy nodes
- one_way: add pairs of nodes incident on the single lanes in the graph (ex: '0 1 5 10' in grid_5_5 graph would convey that the edges between nodes 0-1 and 5-10 are single lane)