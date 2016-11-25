import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np



def finalize(save):
    """Finalize plot
    
    Parameters
    ----------
    save : None or str
        If equal to None then show plot. If is a string containing a path to a
        filename then save plot to this location and close it.
    
    """
    if save is None:
        plt.show()
    else:
        plt.savefig(save)
        plt.close()
    return


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
    # Finalize plot
    finalize(save)
    # Return
    return


def plot_confusion(conf_mat, save=None):
    """Make confusion plot
    
    Parameters
    ----------
    conf_mat : list of array-like
        List of confusion matrices, one for each cross-validation split.
    save : None or string [default None]
        If not equal to None, a string containing a path to a filename in which
        the figure will be saved.
    
    """
    n_splits = len(conf_mat)
    x = np.arange(0, n_splits)
    x = x + 1
    tnp = np.array([100.0 * float(cm[1, 1]) / float(np.sum(cm)) for cm in conf_mat])
    fpp = np.array([100.0 * float(cm[1, 0]) / float(np.sum(cm)) for cm in conf_mat])
    fnp = np.array([100.0 * float(cm[0, 1]) / float(np.sum(cm)) for cm in conf_mat])
    tpp = np.array([100.0 * float(cm[0, 0]) / float(np.sum(cm)) for cm in conf_mat])
    # Set plot parameters
    width = 1.0
    align = 'center'
    xmin = 0.5
    xmax = float(n_splits) + 0.5
    # Make plot
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.bar(x, tnp, color='b', width=width, bottom=None, align=align, label="true negative")
    plt.bar(x, fpp, color='y', width=width, bottom=tnp, align=align, label="false positive")
    plt.bar(x, fnp, color='r', width=width, bottom=tnp + fpp, align=align, label="false negative")
    plt.bar(x, tpp, color='g', width=width, bottom=tnp + fpp + fnp, align=align, label="true positive")
    plt.xlim(xmin, xmax)
    plt.ylim(0.0, 100.0)
    plt.xlabel("split")
    plt.ylabel("proportion (%)")
    plt.legend(loc="best")
    # Finalize plot
    finalize(save)
    return
