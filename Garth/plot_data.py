import os
import sys

def plot_data(prefix):
    'Group and plot the data from the dataset given by prefix.'
    # Get datasets file based on prefix 
    data_set = [f for f in os.listdir('.') if os.path.isfile(f) and
            prefix in str(f) and os.path.splitext(f)[-1] == '.txt']

    # The goal is to produce plot comparing columns values across k

def main(argv):
    'Run the plot with prefix specified by command line argument.'
    if len(sys.argv) == 2:
        plot_data(argv[1])
    else:
        exit(1)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
