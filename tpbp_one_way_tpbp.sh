xterm -e "roscore" &
sleep 3
xterm -e "rosparam load $1" &
sleep 2
xterm -e "rosrun mrpp_sumo sumo_wrapper_one_way.py" &
sleep 3
xterm -e "rosrun mrpp_sumo tpbp_final.py" &
sleep 2
xterm -e "rosrun mrpp_sumo command_center.py" 
sleep 3
killall xterm & sleep 5
