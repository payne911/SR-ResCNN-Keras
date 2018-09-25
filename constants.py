############################
##       UTILS (img)      ##
############################
y_data_path = 'pictures/HR/512/'  # TODO: modify
img_width   = 512
img_height  = 512
augment_im  = True


############################
##         MODEL          ##
############################
img_depth  = 3
scale_fact = 4
res_blocks = 3  # a power of 2, minus 1


############################
##        TRAINING        ##
############################
epochs     = 8
batch_size = 1  # 32 images -> 32 batches
