import numpy as np
import matplotlib.pyplot as plt


class Recording():
    def __init__(self, *args):
        """Set up a dictionary 'metrics' (as parameter of class members) with various recordings, each of which is a list of values across time.
        
        Args:
            An arbitrary number of strings, each of which specifies a key in the metrics dictionary.

        """

        self.metrics = {}
        for arg in args:
            self.metrics[arg] = []

    def add_run(self):
        """Add empty list to metrics dictionary, to be filled during epochs of new run.

        """
        for keyIt in self.metrics:
            self.metrics[keyIt].append([])

    def update(self, **kwargs):
        """Append new measurement.

        Args:
            An arbitrarily number of keyword arguments and respective values.

        """
        
        for key in kwargs:
            assert key in self.metrics, f'Key {key} not found in metrics dictionary.'
            assert len(self.metrics[key]) != 0, 'Add a new run to the recording before beginning to update.'
            #assert isinstance(kwargs[key], np.float32) or isinstance(kwargs[key], int), f'Value for key {key} must be float or int but is {type(kwargs[key])}' # New. Remove in case of issues.
            self.metrics[key][-1].append(kwargs[key])

    def compute_mean_std(self):
        """Compute mean and standard deviation for each metric.
    
        Returns:
            mean_std_dict (dict): Keys are metrics, values are lists of tuples (one for each epoch): (mean, std).

        """
        mean_std_dict  = dict.fromkeys(list(self.metrics.keys()))
        
        for keyIt in self.metrics:
            n_epochs = len(self.metrics[keyIt][0])
            mean_std_dict[keyIt] = np.zeros((n_epochs, 2)) 
            metrics = np.array(self.metrics[keyIt]) # convert to numpy array of shape (n_runs, n_epochs)

            # Now compute mean and stddev
            mean_std_dict[keyIt][:, 0] = np.mean(metrics, axis=0).tolist()
            mean_std_dict[keyIt][:, 1] = np.std(metrics, axis=0).tolist()
        return mean_std_dict

    def plot(self):
        """Plot all metrics over the recorded time.

        """
        n_plots = len(self.metrics)
        if 'timestamp' in self.metrics:
            x_values = self.metrics['timestamp']
        else:
            x_values = range(len(self.metrics[metric]))
        metrics_to_plot = list(self.metrics)
        if 'timestamp' in metrics_to_plot:
            metrics_to_plot.remove('timestamp')
        for i, metric in enumerate(metrics_to_plot):
            ax = plt.subplot(n_plots, 1, i + 1)
            ax.plot(x_values, self.metrics[metric])
            ax.set_ylabel(metric)
        