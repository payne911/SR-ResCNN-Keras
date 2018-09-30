import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-apath', '--absolute_path', type=str, default='./',
                    help="""an absolute path to the image you want to run through the network.
                         For example: C:/Users/PC/Pictures/""")
parser.add_argument('-rpath', '--relative_path', type=str, default='./input/',
                    help="""a relative path to the image you want to run through the network.
                         For example: `input` will result in looking for the image in the folder `input`
                         of this current project (which is actually the default)""")
parser.add_argument('-amnt', '--amount', type=int, default=1,
                    help='how many (cropped to 128x128) samples to predict from within the image')
parser.add_argument('image', type=str, default="example.png",
                    help='image name (example: "bird.png")')

args = parser.parse_args()
# action="store_true"       : if var is in args, set to TRUE, else, set to FALSE

# TODO: redo picture 1 and 2 in "input"

def predict(args):
    print(args.amount)


if __name__ == '__main__':
    print(args)
    predict(args)
