import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np



def plot_learning_curve(train_sizes, train_scores, test_scores, save=None):
    """Make a learning curve plot
    
    Parameters
    ----------
    train_sizes : array-like
        Numbers of training samples that has been used to generate the learning
        curve.
    train_scores : array-like
        Scores on training sets.
    test_scores : array-like
        Scores on test set.
    save : None or string [default None]
        If not equal to None, a string containing a path to a filename in which
        the figure will be saved.
    
    """
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    # Set plot parameters
    train_color = 'b'
    test_color = 'g'
    train_label = "training"
    test_label = "test"
    alpha = 0.2
    xmin = min(train_sizes)
    xmax = max(train_sizes)
    # Make plot
    train_patch = mpatches.Patch(color=train_color, label=train_label)
    test_patch = mpatches.Patch(color=test_color, label=test_label)
    plt.figure()
    plt.fill_between(train_sizes,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     color=train_color,
                     alpha=alpha)
    plt.fill_between(train_sizes,
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     color=test_color,
                     alpha=alpha)
    plt.plot(train_sizes, train_scores_mean, 'o-',
             color=train_color)
    plt.plot(train_sizes, test_scores_mean, 'o-',
             color=test_color)
    plt.grid(True)
    plt.xlim(xmin, xmax)
    plt.xlabel("number of training samples")
    plt.ylabel("score")
    plt.legend(handles=[train_patch, test_patch], loc="best")
    # Show or save plot
    if save is None:
        plt.show()
    else:
        plt.savefig(save)
        plt.close()
    # Return
    return
