import pickle 

filename = "bounce-images_100-shot.p"

tasks = pickle.load(open(filename, "rb"))

new_file = "bounce-images_100-shot-2.p"

pickle.dump(tasks, open(new_file, "wb"), protocol=2)
