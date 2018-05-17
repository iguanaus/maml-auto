import pickle 



def parseSmall():

	filename = "bounce-images_100-shot.p"

	tasks = pickle.load(open(filename, "rb"))

	new_file = "bounce-images_100-shot-2.p"

	pickle.dump(tasks, open(new_file, "wb"), protocol=2)

def parseLarge():
	
	filename = "bounce-images_100-shot.p"

	tasks = pickle.load(open(filename, "rb"))
	bytes_out = pickle.dumps(tasks, protocol=2)
	
	new_file = "bounce-images_100-shot-2.p"


	file_path = new_file
	#bytes_out = pickle.dumps(tasks, protocol=2)
	n_bytes = 2**31
	max_bytes = 2**31 - 1

	with open(file_path, 'wb') as f_out:
	    for idx in range(0, n_bytes, max_bytes):
	        f_out.write(bytes_out[idx:idx+max_bytes])
	        
