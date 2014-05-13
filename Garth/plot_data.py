'''
Simple line plots from data produced by scripts in this repository. Data files
obey the following convention:
    name = prefix[_k].txt where k is optional polynomial order.

    content = mesh_size (x axis variable) in the first column, the other
    quantities of interest (y axis variables) in the other columns.
'''

import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np
import sys
import os
import re

# Data in 2, 3, 4 columns are for error of velocity, pressure and divergence
ylabels = ['$||u-u_h||_{L^2}$',
           '$||p-p_h||_{L^2}$',
           '$||div u_h||_{L^2}$']


def plot_data(prefix, ylabels=ylabels):
    '''Group and plot the data from the dataset given by prefix'''
    # Get datasets file based on prefix
    data_set = [f for f in os.listdir('.') if os.path.isfile(f) and
                prefix in str(f) and os.path.splitext(f)[-1] == '.txt']

    # Files in data set should be prefix_INT.txt. Count files in the dataset
    # FIXME the assumption here is that the order is nonnegative
    ks = [re.match(r'[0-9]+', os.path.splitext(f)[0].split('_')[-1])
          for f in data_set]
    num_ks = len(filter(bool, ks))

    if num_ks:
        ks = map(lambda mo: mo.string, ks)

        # Load the data
        assert len(data_set) == len(ks)
        data = [np.loadtxt(f) for f in data_set]

        n_cols = list(set(d.shape[1] for d in data))
        assert len(n_cols) == 1
        n_cols = n_cols[0]

        for col, ylabel in zip(range(1, n_cols), ylabels):
            fig = plt.figure()
            fig.suptitle(prefix)

            for d, k in zip(data, ks):
                plt.loglog(d[:, 0], d[:, col], '*-', label=k)

                A = np.vstack([np.log(d[:, 0]),
                                  np.ones(len(d[:, 0]))]).T
                print A
                print la.lstsq(A, np.log(d[:, col]))[0]

            plt.xlabel('$h$')
            plt.ylabel(ylabel)
            plt.axis('tight')
            plt.legend(loc='best')



    # Plot single file with data
    else:
        data = np.loadtxt(data_set[0])
        n_rows, n_cols = data.shape

        assert (n_cols - 1) == len(ylabels)

        for col, ylabel in zip(range(1, n_cols), ylabels):
            fig = plt.figure()
            fig.suptitle(prefix)
            plt.loglog(data[:, 0], data[:, col], '*-')
            plt.xlabel('$h$')
            plt.ylabel(ylabel)
            plt.axis('tight')


    plt.show()

def main(argv):
    'Plot dataset with prefix specified by command line argument.'
    if len(sys.argv) == 2:
        plot_data(argv[1])
    else:
        print 'Specify the dataset prefix only.'
        exit(1)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
