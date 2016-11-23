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



class BEERClassifier(BaseEstimator, ClassifierMixin):
    
    
    def __init__(self, params, chan=None, class_weight=None):
        """Initialize the model
        
        Parameters
        ----------
        params : ?
            SpyKING CIRCUS parameters.
        chan : int [default None]
            Index of the channel on which to center.
        class_weight : dict [default None]
            Weights associated with classes. If equal to None, all classes are
            supposed to have weight one.
        
        """
        # Store SpyKING CIRCUS parameters
        self.params_ = params
        # Store central channel parameter
        self.chan_ = chan
        # Store class weights parameter
        self.class_weight_ = class_weight
        # Compute and store various internal parameters
        self.nodes_, self.neighs_ = get_neighbors(self.params_, chan=self.chan_)
        self.src_ = self.chan_ # TODO: correct value...
        self.auto_align_ = False
        self.poly_ = PolynomialFeatures(degree=2,
                                        include_bias=False)
        self.sgd_ = SGDClassifier(loss='log',
                                  fit_intercept=True,
                                  random_state=2,
                                  learning_rate='optimal',
                                  eta0=sys.float_info.epsilon,
                                  class_weight=self.class_weight_)
        self.n_linear_features_ = None
        self.std_scl_ = StandardScaler()
        self.pca_ = PCA(n_components=2)
    
    
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
        
        # TODO: train the BEER classifier...
        # TODO: move parameter settings in __init__ method...
        self.sgd_.set_params(n_iter=1)
        self.sgd_.set_params(eta0=sys.float_info.epsilon)
        self.sgd_.set_params(warm_start=False)
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
    
    
    def score_plot(self, Z, y, sample_weight=None):
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
        matplotlib.pyplot.figure()
        mask_ = numpy.logical_and(y == 0.0, y_pred == 0.0)
        l_0_0 = matplotlib.pyplot.scatter(X_[mask_, 0], X_[mask_, 1], c='g')
        mask_ = numpy.logical_and(y == 0.0, y_pred == 1.0)
        l_0_1 = matplotlib.pyplot.scatter(X_[mask_, 0], X_[mask_, 1], c='y')
        mask_ = numpy.logical_and(y == 1.0, y_pred == 0.0)
        l_1_0 = matplotlib.pyplot.scatter(X_[mask_, 0], X_[mask_, 1], c='r')
        mask_ = numpy.logical_and(y == 1.0, y_pred == 1.0)
        l_1_1 = matplotlib.pyplot.scatter(X_[mask_, 0], X_[mask_, 1], c='b')
        matplotlib.pyplot.legend((l_0_0, l_0_1, l_1_0, l_1_1),
                                 ("hit", "miss", "false alarm", "correct rejection"))
        matplotlib.pyplot.xlabel("first principal component")
        matplotlib.pyplot.ylabel("second principal component")
        matplotlib.pyplot.show()
        
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
            X = get_stas_memshared(self.params_, Z, labels_i, self.src_, self.neighs_,
                                   nodes=self.nodes_, auto_align=self.auto_align_)
        else:
            X = get_stas(self.params_, Z, labels_i, self.src_, self.neighs_,
                         nodes=self.nodes_, auto_align=self.auto_align_)
        
        # Load the PCA basis
        basis_proj, basis_rec = load_data(self.params_, 'basis')
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
