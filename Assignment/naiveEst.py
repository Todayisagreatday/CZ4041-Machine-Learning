# There should be grading rubrics released for this assignment. 
import numpy as np

# Open data.txt
f = open("data.txt", "r")
first_line = f.readline()

# Read first line and save dimensions of data as variables
row, d = first_line.split(',')
row = int(row)
d = int(d)

# Read remaining lines 
array = f.readlines()[0:row]

# Format variable to array 
new_array = []
for line in array: 
    new_array.append((line.strip().split()))

new_array = np.array(new_array, dtype='float32')

def hypercube_kernel(h, x, x_i):
    """
    Implementation of a hypercube kernel for Parzen-window estimation.

    Keyword arguments:
        h: window width
        x: point x for density estimation, 'd x 1'-dimensional numpy array
        x_i: point from training sample, 'd x 1'-dimensional numpy array

    Returns a 'd x 1'-dimensional numpy array as input for a window function.

    """
    assert (x.shape == x_i.shape), 'vectors x and x_i must have the same dimensions'
    return (x - x_i) / (h)


def parzen_window_func(x_vec, h=1):
    """
    Implementation of the window function. Returns 1 if 'd x 1'-sample vector
    lies within inside the window, 0 otherwise.

    """
    for row in x_vec:
        if np.abs(row) > (1/2):
            return 0
    return 1


def parzen_estimation(x_samples, point_x, h, d, window_func, kernel_func):
    """
    Implementation of a parzen-window estimation.

    Keyword arguments:
        x_samples: A 'n x d'-dimensional numpy array, where each sample
            is stored in a separate row. (= training sample)
        point_x: point x for density estimation, 'd x 1'-dimensional numpy array
        h: window width
        d: dimensions
        window_func: a Parzen window function (phi)
        kernel_function: A hypercube or Gaussian kernel functions

    Returns the density estimate p(x).

    """
    k_n = 0
    for row in x_samples:
        x_i = kernel_func(h=h, x=point_x, x_i=row[:,np.newaxis])
        k_n += window_func(x_i, h=h)
    return (k_n / len(x_samples)) / (h**d)

# Print estimation results and save them in output.txt
for i in range(row):
    print(parzen_estimation(new_array, new_array[i].reshape(d,1), h=2, d=d,
                                 window_func=parzen_window_func,
                                 kernel_func=hypercube_kernel), file=open("output.txt", "a"))
