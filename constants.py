############################
##        PREPARE         ##
############################
y_data_path = 'dataset/DIV2K/DIV2K/to_merge/1/'  # Path from where the "y_train" data will be loaded (512x512x3 images)
hr_img_path = 'dataset/DIV2K/DIV2K/DIV2K_train_HR/'  # Path where the "y_train" data will be extracted from (if `prepare_img` is set to True)
crops_p_img = 7      # Number of samples/crops taken per HR image (to get the target output size)
augment_img = True   # Augment data with flips (each image will generate 3 more images)
prepare_img = False  # True => generate cropped images from HR (uses the paths set just above)
# # Deprecated: (used for mini tests)
# y_data_path = 'pictures/HR/512/'
# hr_img_path = 'pictures/HR/'


############################
##       SAVE/LOAD        ##
############################
load_model  = True  # Should we load a saved model from memory ?
save_dir    = 'save'  # folder name where the model will be saved
model_name  = 'my_full_model.h5'  # Name of the model that is to be loaded/saved
# TODO:
model_json  = 'model_architecture.json'
weights     = 'model_weights.h5'


def get_model_save_path():
    return save_dir + '/' + model_name


############################
##         MODEL          ##
############################
img_width  = 512     # size of the output size the network will be trained for
img_height = 512     # this size divided by the scale_fact is the input size of the network
img_depth  = 3    # number of channels (RGB)
scale_fact = 4    # resolution multiplication factor
res_blocks = 3    # a power of 2, minus 1


############################
##        TRAINING        ##
############################
epochs     = 6  # 6 works well
batch_size = 1  # adjust based on your GPU memory (maximize memory usage)
verbosity  = 2  # message feedback from Keras (0, 1 or 2): higher means more verbose
