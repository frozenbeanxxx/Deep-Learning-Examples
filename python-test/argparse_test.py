import os 
import argparse 

def arg_parser():
    parser = argparse.ArgumentParser(description='Darknet To Keras Converter.')
    parser.add_argument('config_path', help='Path to Darknet cfg file.')
    parser.add_argument('weights_path', help='Path to Darknet weights file.')
    parser.add_argument('output_path', help='Path to output Keras model file.')
    parser.add_argument(
        '-p',
        '--plot_model',
        help='Plot generated Keras model and save as image.',
        action='store_true')
    parser.add_argument(
        '-n',
        '--nargs',
        help='Plot generated Keras model and save as image.',
        type=str,
        nargs='+')
    parser.add_argument(
        '-sp',
        '--shape',
        help='Plot generated Keras model and save as image.',
        type=tuple)
    parser.add_argument(
        '-c',
        '--count',
        help='Plot generated Keras model and save as image.',
        #action='store_true',action='store_true'
        type=int,#
        default=200)
    parser.add_argument(
        '-r',
        '--required',
        help='Plot generated Keras model and save as image.',
        required=True)
    parser.add_argument(
        '-i',
        '--integers',
        metavar='N',
        type=int,
        nargs='+',
        help='an integer for the accumulator') 
    parser.add_argument(
        '--sum',
        dest='accumulate',
        action='store_const',
        const=sum,default=min,
        help='sum the integers(default:find the max)')

    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parser()
    print(args)
    print(args.accumulate(args.integers))
    print(os.path.expanduser('~/cython-test/hello.c'))