import os 
for i in range(1,181):
	os.system("python3 sim_visualize.py without_intent_reactive_flag"+str(i))
	print(i, "done without_intent_reactive_flag"+str(i))
