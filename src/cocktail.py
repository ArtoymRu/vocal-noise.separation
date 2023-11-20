import numpy as np
from scipy.io import wavfile as wf

def stereo_to_mono(audio):
    """
    Convert a stereo audio file to mono by averaging the two channels.

    :param audio: A numpy array representing the stereo audio signal.
    :return: A numpy array representing the mono audio signal.
    """
    return np.mean(audio, axis=1)

def center(audio):
    """
    Center an audio signal by subtracting its mean.

    :param audio: A numpy array representing the audio signal.
    :return: A centered numpy array of the audio signal.
    """
    return audio - np.mean(audio)

def create_dummy_signal(audio):
    """
    Create a dummy signal to pair with the audio signal for ICA processing. 
    The dummy signal is a random array of the same shape as the audio signal.

    :param audio: A numpy array representing the audio signal.
    :return: A two-dimensional numpy array with the audio and dummy signals.
    """
    dummy_signal = np.random.random(audio.shape)
    return np.vstack([audio, dummy_signal])

def whiten(signals):
    """
    Whiten the given audio signals, transforming them to be uncorrelated and have unit variance.

    :param signals: A two-dimensional numpy array where each row represents a signal.
    :return: The whitened signals as a two-dimensional numpy array.
    """
    cov_matrix = np.cov(signals)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigen_values))
    signals_whitened = np.dot(eigen_vectors, np.dot(D_inv_sqrt, np.dot(eigen_vectors.T, signals)))
    return signals_whitened


def objFunc(x):
    """
    Objective function for the ICA algorithm.

    :param x: Input data.
    :return: Output of the objective function.
    """
    return np.tanh(x)

def dObjFunc(x):
    """
    Derivative of the objective function for the ICA algorithm.

    :param x: Input data.
    :return: Derivative of the objective function.
    """
    return 1 - np.square(objFunc(x))

def calc_w_hat(W, X):
    """
    Calculate the new value of w in the ICA algorithm.

    :param W: Current value of the weight vector.
    :param X: Input data.
    :return: Updated value of the weight vector.
    """
    w_hat = np.mean(X * objFunc(np.dot(W.T, X)), axis=1) - np.mean(dObjFunc(np.dot(W.T, X))) * W
    w_hat /= np.sqrt(np.sum(np.square(w_hat)))
    return w_hat

def ica(X, iterations, tolerance=1e-5):
    """
    Perform Independent Component Analysis (ICA) on the given data.

    :param X: Input data, a two-dimensional numpy array.
    :param iterations: Number of iterations to run the algorithm.
    :param tolerance: Tolerance for convergence.
    :return: Separated signals as a two-dimensional numpy array.
    """
    num_components = X.shape[0]
    W = np.zeros((num_components, num_components), dtype=X.dtype)
    
    for i in range(num_components):
        w = np.random.rand(num_components)
        for j in range(iterations):
            w_new = calc_w_hat(w, X)
            if i >= 1:
                w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])
            distance = np.abs(np.dot(w_new, w) - 1)
            w = w_new
            if distance < tolerance:
                break
        W[i, :] = w
    
    S = np.dot(W, X)
    return S