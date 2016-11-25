import matplotlib.pyplot
import numpy
import sys

from circus.shared.files import get_stas, get_stas_memshared, load_data
from circus.shared.mpi import SHARED_MEMORY, comm
from circus.validating.utils import get_neighbors

try:
    import sklearn
except Exception:
    if comm.rank == 0:
        print "Sklearn is not installed! Install spyking-circus with the beer extension (see documentation)"
    sys.exit(1)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold



class BEERClassifier(BaseEstimator, ClassifierMixin):
    
    
    def __init__(self, params, chan=None, n_iter=5, class_weight=None):
        """Initialize the model
        
        Parameters
        ----------
        params : ?
            SpyKING CIRCUS parameters.
        chan : int [default None]
            Index of the channel on which to center.
        n_iter : int [default 5]
            The number of passes over the training data (a.k.a. epochs).
        class_weight : dict [default None]
            Weights associated with classes. If equal to None, all classes are
            supposed to have weight one.
        
        """
        # Store SpyKING CIRCUS parameters
        self.params = params
        # Store central channel parameter
        self.chan = chan
        # Store number of passes parameter
        self.n_iter = n_iter
        # Store class weights parameter
        self.class_weight = class_weight
    
    
    def fit(self, Z, y):
        """Fit the model using Z as training spike times and y as target values
        
        Parameters
        ----------
        Z : array-like
            Training spike times of shape (n_samples,).
        
        y : array-like
            Target values (presence or absence of a spike of a chosen neuron) of
            shape (n_samples,).
        
        Returns
        -------
        clf : circus.validating.BEERClassifier
            Current BEER classifier.
        
        """
        # Compute and store various internal parameters
        self.nodes_, self.neighs_ = get_neighbors(self.params, chan=self.chan)
        self.src_ = self.chan # TODO: correct value...
        self.auto_align_ = False
        self.poly_ = PolynomialFeatures(degree=2,
                                        include_bias=False)
        self.sgd_ = SGDClassifier(loss='log',
                                  fit_intercept=True,
                                  n_iter=self.n_iter,
                                  random_state=2,
                                  learning_rate='optimal',
                                  eta0=sys.float_info.epsilon,
                                  class_weight=self.class_weight,
                                  warm_start=False)
        self.n_linear_features_ = None
        self.std_scl_ = StandardScaler()
        self.pca_ = PCA(n_components=2)
        
        # Check that Z and y have correct shape
        assert(Z.ndim == 1)
        assert(y.ndim == 1)
        assert(Z.shape[0] == y.shape[0])
        # Extract snippets identified by Z
        X = self.load_data_(Z, fit_mode=True)
        
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the training spike times
        self.Z_ = Z
        # Store the training spike snippets
        self.X_ = X
        # Store the target values
        self.y_ = y
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # TODO: compute the coefficients and intercept for initialization...
        self.coef_init_ = None
        self.intercept_init_ = None
        # Train the stochastic gradient descent classifier
        self.sgd_.fit(X, y, coef_init=self.coef_init_, intercept_init=self.intercept_init_)
        
        # Return the classifier
        return self
    
    
    def predict(self, Z):
        """Predict target values using the model and Z as spike times
        
        Parameters
        ----------
        Z : array-like
            Training spike times of shape (n_samples,).
        
        Returns
        -------
        y : array-like
            Predicted target values of shape (n_samples,).
        
        """
        # Check is fit had been called
        check_is_fitted(self, ['Z_', 'X_', 'y_'])
        # Input validation
        assert(Z.ndim == 1)
        # Extract snippets identified by Z
        X = self.load_data_(Z, fit_mode=False)
        
        # Predict target values for samples in X
        y = self.sgd_.predict(X)
        
        # Return predicted target values
        return y
    

    def predict_plot(self, Z):
        """TODO
        
        """
        # Check is fit had been called
        check_is_fitted(self, ['Z_', 'X_', 'y_'])
        # Input validation
        assert(Z.ndim == 1)
        # Extract snippets identified by Z
        X = self.load_data_(Z, fit_mode=False)
        
        # Predict target values for samples in X
        y = self.sgd_.predict(X)
        
        # Plot prediction
        X_ = self.get_linear_features_(X)
        X_ = self.std_scl_.transform(X_)
        X_ = self.pca_.transform(X_)
        matplotlib.pyplot.figure()
        mask_ = y == 0.0
        l_0 = matplotlib.pyplot.scatter(X_[mask_, 0], X_[mask_, 1], c='g')
        mask_ = y == 1.0
        l_1 = matplotlib.pyplot.scatter(X_[mask_, 0], X_[mask_, 1], c='r')
        matplotlib.pyplot.legend((l_0, l_1),
                                 ("spike", "no spike"))
        matplotlib.pyplot.xlabel("first principal component")
        matplotlib.pyplot.ylabel("second principal component")
        matplotlib.pyplot.show()
        
        # Return predicted target values
        return y
    
    
    def score_plot(self, Z, y, sample_weight=None, save=None):
        """TODO
        
        """
        # Check is fit had been called
        check_is_fitted(self, ['Z_', 'X_', 'y_'])
        # Input validation
        assert(Z.ndim == 1)
        # Extract snippets identified by Z
        X = self.load_data_(Z, fit_mode=False)
        
        # Predict target values for samples in X
        y_pred = self.sgd_.predict(X)
        
        # Plot prediction
        X_ = self.get_linear_features_(X)
        X_ = self.std_scl_.transform(X_)
        X_ = self.pca_.transform(X_)
        n_samples = X_.shape[0]
        s = 5
        lw = 0.1
        matplotlib.pyplot.figure()
        ax = matplotlib.pyplot.subplot(2, 2, 1)
        mask_ = numpy.logical_and(y == 0.0, y_pred == 0.0)
        matplotlib.pyplot.scatter(X_[mask_, 0], X_[mask_, 1], c='g', s=s, lw=lw)
        n_hits = numpy.sum(mask_)
        p_hits = 100.0 * float(n_hits) / float(n_samples)
        matplotlib.pyplot.title("{} hits ({:.2f}%)".format(n_hits, p_hits))
        matplotlib.pyplot.xlabel("first principal component")
        matplotlib.pyplot.ylabel("second principal component")
        matplotlib.pyplot.subplot(2, 2, 2, sharex=ax, sharey=ax)
        mask_ = numpy.logical_and(y == 0.0, y_pred == 1.0)
        matplotlib.pyplot.scatter(X_[mask_, 0], X_[mask_, 1], c='r', s=s, lw=lw)
        n_misses = numpy.sum(mask_)
        p_misses = 100.0 * float(n_misses) / float(n_samples)
        matplotlib.pyplot.title("{} misses ({:.2f}%)".format(n_misses, p_misses))
        matplotlib.pyplot.xlabel("first principal component")
        matplotlib.pyplot.ylabel("second principal component")
        matplotlib.pyplot.subplot(2, 2, 3, sharex=ax, sharey=ax)
        mask_ = numpy.logical_and(y == 1.0, y_pred == 0.0)
        matplotlib.pyplot.scatter(X_[mask_, 0], X_[mask_, 1], c='r', s=s, lw=lw)
        n_false_alarms = numpy.sum(mask_)
        p_false_alarms = 100.0 * float(n_false_alarms) / float(n_samples)
        matplotlib.pyplot.title("{} false alarms ({:.2f}%)".format(n_false_alarms, p_false_alarms))
        matplotlib.pyplot.xlabel("first principal component")
        matplotlib.pyplot.ylabel("second principal component")
        matplotlib.pyplot.subplot(2, 2, 4, sharex=ax, sharey=ax)
        mask_ = numpy.logical_and(y == 1.0, y_pred == 1.0)
        matplotlib.pyplot.scatter(X_[mask_, 0], X_[mask_, 1], c='g', s=s, lw=lw)
        n_correct_rejections = numpy.sum(mask_)
        p_correct_rejections = 100.0 * float(n_correct_rejections) / float(n_samples)
        matplotlib.pyplot.title("{} correct rejections ({:.2f}%)".format(n_correct_rejections, p_correct_rejections))
        matplotlib.pyplot.xlabel("first principal component")
        matplotlib.pyplot.ylabel("second principal component")
        matplotlib.pyplot.tight_layout()
        if save is None:
            matplotlib.pyplot.show()
        else:
            matplotlib.pyplot.savefig(save)
            matplotlib.pyplot.close()
        
        score = accuracy_score(y, y_pred, sample_weight=sample_weight)
        return score
    
    
    def load_data_(self, Z, fit_mode=True):
        """Load the data/snippets
        
        Parameters
        ----------
        Z : array-like
            Spike times of shape (n_samples,).
        fit_mode : bool [default True]
            If True then the preprocessing steps are fitted with the input data.
        
        Returns
        -------
        X : array-like
            Snippets of shape (n_samples, n_electrodes * w_template).
        
        """
        # Extract the snippets
        labels_i = numpy.zeros_like(Z) # TODO: modifiy 'get_stas_memshared' and 'get_stas' in order to be able remove 'labels_i'...
        if SHARED_MEMORY:
            X = get_stas_memshared(self.params, Z, labels_i, self.src_, self.neighs_,
                                   nodes=self.nodes_, auto_align=self.auto_align_)
        else:
            X = get_stas(self.params, Z, labels_i, self.src_, self.neighs_,
                         nodes=self.nodes_, auto_align=self.auto_align_)
        
        # Load the PCA basis
        basis_proj, basis_rec = load_data(self.params, 'basis')
        # Retrieve various sizes
        n_samples = X.shape[0]
        n_channels = X.shape[1]
        n_timesteps = X.shape[2]
        n_features = basis_proj.shape[1]
        # Reshape snippets
        X = X.reshape((n_samples * n_channels, n_timesteps))
        # Project snippets on the PCA basis
        X = numpy.dot(X, basis_proj)
        # Reshape projected snippets
        X = X.reshape((n_samples, n_channels * n_features))
        
        # TODO: center and reduce snippets before generating polynomial features...
        if fit_mode:
            # Fit polynomial features generator
            self.poly_.fit(X)
        # Generate polynomial features
        X = self.poly_.transform(X)
        
        if fit_mode:
            # Fit principal component analysis
            self.n_linear_features_ = n_channels * n_features
            X_ = self.get_linear_features_(X)
            self.std_scl_.fit(X_)
            X_ = self.std_scl_.transform(X_)
            self.pca_.fit(X_)
        
        # Return loaded data
        return X
    
    
    def get_linear_features_(self, X):
        """TODO
        """
        assert(self.n_linear_features_ is not None)
        X = X[:, 0:self.n_linear_features_]
        return X
    
    
    def fit_(self):
        """Fit the neural network
        
        Returns
        -------
        clf : circus.validating.BEERClassifier
            Current BEER classifier.
        """
        return self



class BEERPredictor(object):
    
    
    def __init__(self, params, n_splits=3, shuffle=False, chan=None, n_iter=5, class_weight=None):
        """Initialize the model
        
        Parameters
        ----------
        params : ?
            SpyKING CIRCUS parameters.
        n_splits : int [default 3]
            Number of folds. Must be at least 2.
        
        TODO: complete...
        
        """
        # Store SpyKING CIRCUS parameters
        self.params = params
        # Store number of folds
        self.n_splits = n_splits
        # TODO: comment...
        self.shuffle = shuffle
        self.random_state_ = 22346 # TODO: correct...
        self.chan = chan
        self.class_weight = class_weight
        self.n_iter = n_iter
        # Compute and store various internal parameters
        self.kf_ = KFold(n_splits=self.n_splits,
                         shuffle=self.shuffle,
                         random_state=self.random_state_)
    
    
    def predict(self, Z, y, mode='plain'):
        """TODO: complete...
        
        mode: 'plain' of 'split'
            TODO: complete...
        
        """
        # Check if mode is valid
        assert(mode in ['plain', 'split'])
        
        # Preallocation
        if mode is 'plain':
            y_pred = numpy.empty_like(y)
        elif mode is 'split':
            y_pred = self.n_splits * [None]
        
        # For each fold
        for index, (train_index, test_index) in enumerate(self.kf_.split(Z)):
            Z_train = Z[train_index]
            Z_test = Z[test_index]
            y_train = y[train_index]
            y_test = y[test_index]
            beer = BEERClassifier(self.params,
                                  chan=self.chan,
                                  n_iter=self.n_iter,
                                  class_weight=self.class_weight)
            beer.fit(Z_train, y_train)
            if mode is 'plain':
                y_pred[test_index] = beer.predict(Z_test)
            elif mode is 'split':
                y_pred[index] = beer.predict(Z_test)
        
        # Return predicted target values
        return y_pred
    
    
    def predict_plot(self, Z, y):
        """TODO: complete...
        
        """
        # Preallocation
        y_pred = numpy.empty_like(y)
        
        # For each fold
        for train_index, test_index in self.kf_.split(Z):
            # Retrieve the training data and labels from other folds
            Z_train = Z[train_index]
            y_train = y[train_index]
            # Retrieve the test data and labels from current fold
            Z_test = Z[test_index]
            y_test = y[test_index]
            # Initialize a new BEER classifier
            beer = BEERClassifier(self.params,
                                  chan=self.chan,
                                  n_iter=self.n_iter,
                                  class_weight=self.class_weight)
            # Fit BEER classifier on training data and labels
            beer.fit(Z_train, y_train)
            # Store predicted target values on the test data
            y_pred[test_index] = beer.predict_plot(Z_test)
        
        # Return predicted target values
        return y_pred
    
    
    def score(self, Z, y, sample_weight=None):
        """TODO: complete...
        
        """
        y_pred = self.predict(Z, y)
        score = accuracy_score(y, y_pred, sample_weight=sample_weight)
        return score
    
    
    def score_plot(self, Z, y, sample_weight=None):
        """TODO: complete...
        
        """
        # Preallocation
        scores = numpy.zeros(self.n_splits)
        
        # For each fold
        for index, (train_index, test_index) in enumerate(self.kf_.split(Z)):
            # Retrieve the training data and labels from other folds
            Z_train = Z[train_index]
            y_train = y[train_index]
            # Retrieve the test data and labels from current fold
            Z_test = Z[test_index]
            y_test = y[test_index]
            # Initialize a new BEER classifier
            beer = BEERClassifier(self.params,
                                  chan=self.chan,
                                  n_iter=self.n_iter,
                                  class_weight=self.class_weight)
            # Fit BEER classifier on training data and labels
            beer.fit(Z_train, y_train)
            # Store mean accuracy on the test data and labels
            scores[index] = beer.score_plot(Z_test, y_test)
        
        # Return mean accuracies
        return scores
    
    
    def evaluate(self, Z, y):
        """TODO: complete...
        
        """
        # Preallocation
        conf_mat = self.n_splits * [None]
        y_pred = self.predict(Z, y, mode='split')
        y = self.split_(y)
        for index, (y_fold, y_pred_fold) in enumerate(zip(y, y_pred)):
            conf_mat[index] = confusion_matrix(y_fold, y_pred_fold, labels=[0.0, 1.0])
        return conf_mat
    
    
    def split_(self, y):
        """Split target values into folds
        
        TODO: complete...
        
        """
        # Preallocation
        y_ = self.n_splits * [None]
        # For each fold
        for index, (train_index, test_index) in enumerate(self.kf_.split(y)):
            y_[index] = y[test_index]
        # Return target values in folds
        return y_
