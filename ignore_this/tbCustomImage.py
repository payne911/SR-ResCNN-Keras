# # Related to the failed attempt of displaying custom image in TensorBoard (see "callbacks.py")
# import tensorflow as tf
# from keras.callbacks import Callback
# import io
# from PIL import Image

# tbi_callback = TensorBoardImage('Image test')

# def make_image(tensor):
#     """
#     Convert an numpy representation image to Image protobuf.
#     Copied from https://github.com/lanpa/tensorboard-pytorch/
#     """
#     height, width, channel = tensor.shape
#     image = Image.fromarray(tensor.astype('uint8'))  # TODO: maybe float ?
#
#     output = io.BytesIO()
#     image.save(output, format='JPEG')
#     image_string = output.getvalue()
#     output.close()
#
#     return tf.Summary.Image(height=height,
#                             width=width,
#                             colorspace=channel,
#                             encoded_image_string=image_string)
#
#
# class TensorBoardImage(Callback):
#     def __init__(self, tag):
#         super().__init__()
#         self.tag = tag
#
#     def on_epoch_end(self, epoch, logs={}):
#         # Load image
#         img_input = self.validation_data[0][0]  # X_train
#         img_valid = self.validation_data[1][0]  # Y_train
#
#         print(self.validation_data[0].shape)  # (8, 128, 128, 3)
#         print(self.validation_data[1].shape)  # (8, 512, 512, 3)
#
#         image = make_image(img_input)
#         summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])
#         writer = tf.summary.FileWriter('./logs')
#         writer.add_summary(summary, epoch)
#         writer.close()
#
#         image = make_image(img_valid)
#         summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])
#         writer = tf.summary.FileWriter('./logs')
#         writer.add_summary(summary, epoch)
#         writer.close()
#
#         return
#
#
# class PeriodicLogger(Callback):
#
#     def on_train_begin(self, logs={}):
#         # Initialization code
#         self.epochs = 0
#
#     def on_epoch_end(self, batch, logs={}):
#         self.epochs += 1
#         if self.epochs % 2 == 0:
#             # Do stuff like printing metrics
#             print("bob")
