import argparse
import os

parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Data and model checkpoints directories
parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                    help='data directory containing input.txt with training examples')
parser.add_argument('--save_dir', type=str, default='save',
                    help='directory to store checkpointed models')
parser.add_argument('--log_dir', type=str, default='logs',
                    help='directory to store tensorboard logs')
parser.add_argument('--save_every', type=int, default=1000,
                    help='Save frequency. Number of passes between checkpoints of the model.')
parser.add_argument('--init_from', type=str, default=None,
                    help="""continue training from saved model at this path (usually "save").
                        Path must contain files saved by previous training process:
                        'config.pkl'        : configuration;
                        'chars_vocab.pkl'   : vocabulary definitions;
                        'checkpoint'        : paths to model file(s) (created by tf).
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition (created by tf)
                         Model params must be the same between multiple runs (model, rnn_size, num_layers and seq_length).
                    """)
# Model params
parser.add_argument('--model', type=str, default='lstm',
                    help='lstm, rnn, gru, or nas')
parser.add_argument('--rnn_size', type=int, default=128,
                    help='size of RNN hidden state')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of layers in the RNN')
# Optimization
parser.add_argument('--seq_length', type=int, default=50,
                    help='RNN sequence length. Number of timesteps to unroll for.')
parser.add_argument('--batch_size', type=int, default=75,
                    help="""minibatch size. Number of sequences propagated through the network in parallel.
                            Pick batch-sizes to fully leverage the GPU (e.g. until the memory is filled up)
                            commonly in the range 10-500.""")
parser.add_argument('--num_epochs', type=int, default=50,
                    help='number of epochs. Number of full passes through the training examples.')
parser.add_argument('--grad_clip', type=float, default=5.,
                    help='clip gradients at this value')
parser.add_argument('--learning_rate', type=float, default=0.002,
                    help='learning rate')
parser.add_argument('--decay_rate', type=float, default=0.97,
                    help='decay rate for rmsprop')
parser.add_argument('--output_keep_prob', type=float, default=1.0,
                    help='probability of keeping weights in the hidden layer')
parser.add_argument('--input_keep_prob', type=float, default=1.0,
                    help='probability of keeping weights in the input layer')
args = parser.parse_args()


def train(args):
    print(args)
    # data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    # args.vocab_size = data_loader.vocab_size
    #
    # # check compatibility if training is continued from previously saved model
    # if args.init_from is not None:
    #     # check if all necessary files exist
    #     assert os.path.isdir(args.init_from), " %s must be a a path" % args.init_from
    #     assert os.path.isfile(os.path.join(args.init_from, "config.pkl")), "config.pkl file does not exist in path %s" % args.init_from
    #     assert os.path.isfile(os.path.join(args.init_from, "chars_vocab.pkl")), "chars_vocab.pkl.pkl file does not exist in path %s" % args.init_from
    #     ckpt = tf.train.latest_checkpoint(args.init_from)
    #     assert ckpt, "No checkpoint found"
    #
    #     # open old config and check if models are compatible
    #     with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
    #         saved_model_args = cPickle.load(f)
    #     need_be_same = ["model", "rnn_size", "num_layers", "seq_length"]
    #     for checkme in need_be_same:
    #         assert vars(saved_model_args)[checkme] == vars(args)[checkme], "Command line argument and saved model disagree on '%s' " % checkme
    #
    #     # open saved vocab/dict and check if vocabs/dicts are compatible
    #     with open(os.path.join(args.init_from, 'chars_vocab.pkl'), 'rb') as f:
    #         saved_chars, saved_vocab = cPickle.load(f)
    #     assert saved_chars == data_loader.chars, "Data and loaded model disagree on character set!"
    #     assert saved_vocab == data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"
    #
    # if not os.path.isdir(args.save_dir):
    #     os.makedirs(args.save_dir)
    # with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
    #     cPickle.dump(args, f)
    # with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
    #     cPickle.dump((data_loader.chars, data_loader.vocab), f)

    # model = Model(args)

if __name__ == '__main__':
    train(args)
