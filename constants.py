############################
##        GENERAL         ##
############################
y_data_path = 'pictures/HR/512/'
hr_img_path = 'pictures/HR/'
crops_p_img = 5
img_width   = 512
img_height  = 512
augment_im  = True
prepare_im  = False
verbosity   = 2


############################
##         MODEL          ##
############################
img_depth  = 3
scale_fact = 4
res_blocks = 3  # a power of 2, minus 1


############################
##        TRAINING        ##
############################
epochs     = 6
batch_size = 1  # 32 images -> 32 batches
