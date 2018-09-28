############################
##        PREPARE         ##
############################
y_data_path = 'pictures/HR/512/'
hr_img_path = 'pictures/HR/'
crops_p_img = 5
augment_img = True
prepare_img = False


############################
##       SAVE/LOAD        ##
############################
save_model  = True  # Do you want to save the model once it has run?
model_saved = True  # Should we load a saved model from memory ?
save_dir    = 'save'
model_name  = 'my_model.h5'
# Deprecated:
model_json  = 'model_architecture.json'
weights     = 'model_weights.h5'


def get_model_save_path():
    return save_dir + '/' + model_name


############################
##         MODEL          ##
############################
img_width  = 512
img_height = 512
img_depth  = 3
scale_fact = 4
res_blocks = 3  # a power of 2, minus 1


############################
##        TRAINING        ##
############################
epochs     = 30  # 6 worked well, now testing "ReduceLROnPlateau" Keras callback
batch_size = 1  # 32 images -> 32 batches
verbosity  = 2
