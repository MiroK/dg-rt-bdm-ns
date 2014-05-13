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


def rate_plot(x, y, label=''):
    'Plot log(x) vs. log(y)'
    # Compute rate from least square fit to log(y) = p*log(x) + q
    A = np.vstack([np.log(x), np.ones(len(x))]).T
    p, q = la.lstsq(A, np.log(y))[0]

    # Plot
    if label:
        label = '$%s$,' % label
    plt.loglog(x, y, '*-', label=' '.join([label, '$p=%.2f$' % p]))


class RateFigure(object):
    'Context manager for figure with (multiple) rate plot(s).'
    def __init__(self, prefix, ylabel):
        fig = plt.figure()
        fig.canvas.set_window_title(prefix)
        plt.ylabel(ylabel)
        self.fig = fig

    def __enter__(self):
        return self.fig

    def __exit__(self, type, value, traceback):
        plt.xlabel('$h$')
        plt.axis('tight')
        plt.legend(loc='best')
        plt.show()


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

    # Group `column` plots for all k
    if num_ks:
        ks = map(lambda mo: mo.string, ks)

        # Load the data
        assert len(data_set) == num_ks
        data = [np.loadtxt(f) for f in data_set]

        # Verify that all data in data sets have same number of columns
        n_cols = list(set(d.shape[1] for d in data))
        assert len(n_cols) == 1
        n_cols = n_cols[0]

        # Check that there is label for all but the first column
        assert (n_cols - 1) == len(ylabels)

        # Group for each column
        for col, ylabel in zip(range(1, n_cols), ylabels):
            with RateFigure(prefix, ylabel) as fig:
                for d, k in zip(data, ks):
                    rate_plot(d[:, 0], d[:, col], k)

    # Plot single file with data
    else:
        data = np.loadtxt(data_set[0])
        n_cols = data.shape[1]

        assert (n_cols - 1) == len(ylabels)

        for col, ylabel in zip(range(1, n_cols), ylabels):
            with RateFigure(prefix, ylabel) as fig:
                rate_plot(data[:, 0], data[:, col])


def main(argv):
    'Plot dataset with prefix specified by command line argument.'
    if len(sys.argv) == 2:
        plot_data(argv[1])
    else:
        print 'Specify the dataset prefix only.'
        exit(1)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
