# This file takes in the C-sin-10-shot and converst it into the ball bouncing state data. 
import pickle 
import numpy as np



filename = "C-sin_10-shot_legit_2.p"
#filename = "bounce-states_100-shot_2.p"
new_file = "C-sin_10-shot_legit_stateform.p"

tasks = pickle.load(open(filename, "rb"))

#Now convert it

def restructure(indice="tasks_train"):
	print("number of tasks: " , len(tasks[indice]))
	newList = []
	for j in range(0,len(tasks[indice])):
		if j % 100 == 0:
			print j
			print(float(j)/len(tasks[indice]))
		#Limit to the first 1000
		if j > 999:
			break
		firstTask = tasks[indice][j]
		dataTask = firstTask[0]
		infoTask = firstTask[1]
		traina = dataTask[0]
		trainb = dataTask[1]
		inputa = traina[0]
		labela = traina[1]
		inputb = trainb[0]
		labelb = trainb[1]
		#print('ina shape:',inputa.shape)
		#print(labela.shape)
		#print(inputb.shape)
		#print(labelb.shape)
		ina_new = np.tile(inputa.ravel(),(6,1)).transpose().reshape(-1,3,2)
		laa_new = np.tile(labela.ravel(),(200,1)).transpose().reshape(-1,100,2)

		inb_new = np.tile(inputb.ravel(),(6,1)).transpose().reshape(-1,3,2)
		lab_new = np.tile(labelb.ravel(),(200,1)).transpose().reshape(-1,100,2)

		#print("ina_new: " , ina_new)
		#print("ina_new: " , len(ina_new))
		#os.exit()

		#traina[0] = ina_new
		#traina[1] = laa_new
		#trainb[0] = inb_new
		#trainb[1] = lab_new
		n_dataTask = [[[ina_new,laa_new],[inb_new,lab_new]],infoTask]
		#print(n_dataTask)
		newList.append(n_dataTask)
		#print("Done")
		#break
		#break
	return newList
	#tasks[indice] = newList
def pullVals():
	print("Adding train....")
	l1 = restructure("tasks_train")
	print("Adding test ....")
	l2 = restructure("tasks_test")

	tasks = {'tasks_train':l1,'tasks_test':l2}
	#print("Tasks: ", tasks)



	pickle.dump(tasks, open(new_file, "wb"))


pullVals()






