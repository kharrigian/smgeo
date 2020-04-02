

#######################
### Imports
#######################
 
## Standard Library
import sys

## External Library
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegressionCV
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from global_land_mask import globe # pip install global-land-mask
import matplotlib.pyplot as plt

#######################
### Helpers
#######################
 
def assign_value_to_bin(value, bins):
    """

    """
    b = 0
    for bstart, bend in zip(bins[:-1], bins[1:]):
        if value >= bstart and value < bend:
            return b
        b += 1
    return b

f1_score = metrics.make_scorer(
         metrics.f1_score,
         greater_is_better=True,
         needs_threshold=False,
         average="weighted",
)

#######################
### Inference Model
#######################
 
class GeolocationInference(object):
    
    """

    """

    def __init__(self,
                 vocabulary,
                 n_time_bins=10,
                 time_norm=None,
                 time_as_percentile=True,
                 time_model_kwargs = {"Cs":10, "solver":'lbfgs', "n_jobs":-1},
                 mixture_kwargs = {"n_components":5,"covariance_type":"diag"},
                 random_state = 42):
        """

        """
        ## Class Attributes/Parameters
        self._vocabulary = vocabulary
        self._n_time_bins = n_time_bins
        self._time_norm = time_norm
        self._time_as_percentile = time_as_percentile
        self._time_model_kwargs = time_model_kwargs
        self._mixture_kwargs = mixture_kwargs
        self._random_state = random_state
        ## Check Arguments
        if "random_state" not in self._mixture_kwargs:
            self._mixture_kwargs["random_state"] = self._random_state
        if "random_state" not in self._time_model_kwargs:
            self._time_model_kwargs["random_state"] = self._random_state

    def __repr__(self):
        """

        """
        return "GeolocationInference()"
    
    def _create_coordinate_grid(self,
                                cell_size = 1):
        """

        """
        ## Boundaries
        xmin = int(self._coord_bounds[0][0] - 1)
        xmax = int(self._coord_bounds[0][1] + 1)
        ymin = int(self._coord_bounds[1][0] - 1)
        ymax = int(self._coord_bounds[1][1] + 1)
        ## Coordinates
        lon_coord = [xmin]
        lat_coord = [ymin]
        while lon_coord[-1] < xmax:
            lon_coord.append(min(lon_coord[-1] + cell_size, xmax))
        while lat_coord[-1] < ymax:
            lat_coord.append(min(lat_coord[-1] + cell_size, ymax))
        coordinate_grid = []
        for x in lon_coord:
            for y in lat_coord:
                if globe.is_land(y, x):
                    coordinate_grid.append([x, y])
        coordinate_grid = np.array(coordinate_grid)
        return coordinate_grid
    

    def _fit_mixture(self,
                     i,
                     X,
                     y):
        """

        """
        ## Construct Training Sample
        S = []
        nonzero = np.nonzero(X[:,i])[0]
        for n in nonzero:
            x_s = int(X[n, i])
            y_s = y[n]
            S.extend([y_s] * x_s)
        S = np.vstack(S)
        ## Initialize Model
        n_components = min(self._mixture_kwargs["n_components"], len(nonzero))
        args = self._mixture_kwargs.copy()
        args["n_components"] = n_components
        model = BayesianGaussianMixture(**args)
        ## Fit Model
        model = model.fit(S)
        ## Cache Model
        self._models[i] = model

    def fit(self,
            X,
            y):
        """
        Args:
            X (csr matrix): Sparse feature matrix
            y (2d-array): Training coordinates
        
        Returns:
            self
        """
        ## Coordinate Grid Boundaries
        self._coord_bounds = [[y[:,0].min(), y[:,0].max()], [y[:,1].min(), y[:,1].max()]]
        ## Initialize Mixture Model Cache
        m = len(self._vocabulary._feature_inds["text"]) + \
            len(self._vocabulary._feature_inds["subreddit"])
        self._models = [None for _ in range(m)]
        ## Feature Probabilities
        self._pu = np.array((X > 0).sum(axis=0) / X.shape[0])[0]
        ## Fit Prior
        self.prior = BayesianGaussianMixture(**self._mixture_kwargs)
        self.prior.fit(y)
        ## Fit Mixture Models
        for i in tqdm(range(m), desc="GMM Fit", total = m, file=sys.stdout):
            self._fit_mixture(i, X, y)
        ## Temporal Model
        if self._vocabulary._use_time:
            ## Isolate Time Data
            X_time = X[:, self._vocabulary._feature_inds["time"]]
            ## Normalize
            if self._time_norm is not None:
                X_time = normalize(X_time, self._time_norm, axis=1, copy=True)
            ## Create Time Bins
            y_lon = y[:,0]
            if self._time_as_percentile:
                percentiles = np.linspace(0, 100, self._n_time_bins + 1)[:-1]
                self._time_bins = np.percentile(y_lon, percentiles)
            else:
                lon_bounds = int(y_lon.min() - 1), int(y_lon.max() + 1)
                self._time_bins = np.linspace(lon_bounds[0], lon_bounds[1], self._n_time_bins + 1)
            y_lon = np.array(list(map(lambda v: assign_value_to_bin(v, self._time_bins), y_lon)))
            ## Train Model
            self._time_model_kwargs["scoring"] = f1_score
            self._time_model_kwargs["max_iter"] = 1000
            self.time_classifier = LogisticRegressionCV(**self._time_model_kwargs)
            self.time_classifier.fit(X_time, y_lon)
        return self
        
    def predict_proba(self,
                      X,
                      coordinates = None):
        """
        Args:
            X (csr matrix): Sparse feature matrix
            coordinates (2d-array): Coordinates to make predictions on. If None, creates grid
                                    around the world (land-masses only)
        
        Returns:
            coordinates (2d-array): Coordinate array associated with prediction columns
            P (2d-array): Predicted probabilities per coordinate
        """
        ## Coordinate Grid
        if coordinates is None:
            coordinates = self._create_coordinate_grid()
        ## Transform X Shape
        X = X.toarray()
        ## Compute Prior for Elements With Missing
        prior = np.exp(self.prior.score_samples(coordinates))
        ## Initialize Probability Array
        P = np.zeros((X.shape[0], len(coordinates)))
        ## Cycle Through Models
        for ind, model in tqdm(enumerate(self._models),
                               total = len(self._models),
                               file=sys.stdout,
                               desc="GMM Posterior"):
            if model is None:
                continue
            u = X[:,[ind]]
            nonzero = np.nonzero(u)[0]
            if len(nonzero) == 0:
                continue
            p_c_u = np.exp(model.score_samples(coordinates)).reshape(1,-1)
            P[nonzero] += np.matmul(u[nonzero], p_c_u * self._pu[ind])
        ## Default to Prior for Users Without Features
        P[np.all(P == 0, axis=1)] += prior
        ## Time Adjustment
        if self._vocabulary._use_time:
            ## Isolate Time Data
            X_time = X[:, self._vocabulary._feature_inds["time"]]
            ## Normalize
            if self._time_norm is not None:
                X_time = normalize(X_time, self._time_norm, axis=1, copy=True)
            ## Make Probability Predictions
            y_lon_pred_prob = self.time_classifier.predict_proba(X_time)
            ## Get Time Bins
            time_bins_classifier = self._time_bins[self.time_classifier.classes_]
            coordinate_lon_bins = np.array(list(map(lambda v: assign_value_to_bin(v, self._time_bins),
                                                    coordinates[:,0])))
            ## Update Probabilities
            P_time = y_lon_pred_prob[:, coordinate_lon_bins]
            P_time = normalize(P_time, axis=1, norm="l1")
            P = np.multiply(P, P_time)
        return coordinates, P

    def predict(self,
                X,
                coordinates=None):
        """
        Args:
            X (csr matrix): Feature Matrix
            coordinates (2d-array or None): Coordinates to make predictions over
        
        Returns:
            y_pred (2d-array): Coordinate Predictions (argmax)
        """
        ## Get Probability Prediction
        coordinates, P = self.predict_proba(X, coordinates)
        ## Get Argmax over Coordinates
        y_pred = coordinates[P.argmax(axis=1)]
        return y_pred
    
    def plot_model_posterior(self,
                             feature,
                             coordinates=None):
        """

        """
        ## Check for Feature/Model
        if feature not in self._vocabulary.feature_to_idx:
            raise KeyError(f"Feature=`{feature} not found")
        if self._models[self._vocabulary.feature_to_idx[feature]] is None:
            raise ValueError(f"Model for Feature={feature} is null")
        ## Coordinate Grid
        if coordinates is None:
            coordinates = self._create_coordinate_grid(cell_size=2)
        ## Make Posterior Predictions
        m = self._models[self._vocabulary.feature_to_idx[feature]]
        posterior = np.exp(m.score_samples(coordinates))
        ## Plot 
        fig, ax = plt.subplots(figsize=(10,5.8))
        s = ax.scatter(coordinates[:,0],
                       coordinates[:,1],
                       c = posterior,
                       cmap = plt.cm.coolwarm,
                       alpha = .8,
                       s = 5)
        cbar = fig.colorbar(s)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(feature, loc="left")
        fig.tight_layout()
        return fig, ax