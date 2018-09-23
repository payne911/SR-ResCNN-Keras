############################
##       UTILS (img)      ##
############################
# data_path  = 'pictures/1400/'
# img_width  = 1400
# img_height = 1400
data_path  = 'pictures/512/'
img_width  = 512
img_height = 512


############################
##         MODEL          ##
############################
img_depth  = 3
scale_fact = 4
res_blocks = 1  # a power of 2, minus 1


############################
##        TRAINING        ##
############################
epochs     = 2
batch_size = 1  # 32 images -> 32 batches
