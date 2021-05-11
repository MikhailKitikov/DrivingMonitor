import multiprocessing
import argparse
import dlib


ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,
	help="path to input training XML file")
ap.add_argument("-m", "--model", required=True,
	help="path serialized dlib shape predictor model")
args = vars(ap.parse_args())

print("[INFO] setting shape predictor options...")
options = dlib.shape_predictor_training_options()

options.tree_depth = 4
options.nu = 0.1
options.cascade_depth = 15
options.feature_pool_size = 400
options.num_test_splits = 50
options.oversampling_amount = 5
options.oversampling_translation_jitter = 0.1
options.be_verbose = True
options.num_threads = multiprocessing.cpu_count()

print("[INFO] shape predictor options:")
print(options)

print("[INFO] training shape predictor...")
dlib.train_shape_predictor(args["training"], args["model"], options)
