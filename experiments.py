import os
import sys
import math
import itertools

# import warnings
# import json
import time
import psutil
import socket

# import shelve
# import threading
import traceback
# import inspect
import logging
from multiprocessing import Pool, Queue
from concurrent.futures import ThreadPoolExecutor

# from io import StringIO
from types import SimpleNamespace
from typing import Tuple, Dict, List, Any, Union, TypeVar, Type, Sequence, Generic
from datetime import datetime, timedelta
from dateutil import tz

# from functools import partial
#from dataclasses import dataclass, field
from collections import OrderedDict

import numpy as np
import pandas as pd

# from mpi4py import MPI
import h5py
import humanize

from pprint import pformat

# R-coefficient PEARSON
# from scipy.stats import pearsonr

from skopt import BayesSearchCV
import skopt.space
#from skopt.space import Real, Categorical, Integer

# if you want to custom a estimator
# from sklearn.base import BaseEstimator

from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from sklearn.linear_model import Lasso

from sklearn.metrics import (
    precision_recall_fscore_support,
    f1_score,
    accuracy_score,
    confusion_matrix,
    confusion_matrix,
)
from sklearn.exceptions import UndefinedMetricWarning

from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.decomposition import IncrementalPCA, PCA

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.feature_selection import (
    SequentialFeatureSelector,
    SelectKBest,
    RFE,
    f_classif,
    chi2,
    mutual_info_classif,
)

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# import mlflow
# import mlflow.sklearn

# from hpelm import ELM
#from elm import ELM
from skelm import ELMClassifier

import fire
#import stackprinter
from traceback_with_variables import activate_by_import, print_exc

from utils.libutils import swfe  # pylint: disable-msg=E0611

np.set_printoptions(precision=4, linewidth=120)
pd.set_option("precision", 4)
pd.set_option("display.width", 300)

# should have used this from day 1
# pd.options.mode.use_inf_as_na = True

#@dataclass(frozen=True)
class Results:
    experiment: str
    timestamp: int
    class_: str
    seed: int
    foldoutter: int
    foldinner: int
    classifier: str
    classifiercfg: int
    classifiercfgs: int
    f1binp: float
    f1binn: float
    f1micro: float
    f1macro: float
    f1weighted: float
    f1samples: float
    precision: float
    recall: float
    accuracy: float
    accuracy2: float
    timeinnertrain: float
    timeouttertrain: float
    positiveclasses: str
    negativeclasses: str
    features: str
    nfeaturesvar: int
    nfeaturestotal: int
    ynegfinaltrain: int
    yposfinaltrain: int
    ynegfinaltest: int
    yposfinaltest: int
    yposfinalpred: int
    ynegfinalpred: int
    yfinaltrain: int
    yfinaltest: int
    yfinalpred: int
    postrainsamples: int
    negtrainsamples: int
    postestsamples: int
    negtestsamples: int
    tp: int
    tn: int
    fp: int
    fn: int
    bestfeatureidx: int
    bestvariableidx: int
    featurerank: str  # em ordem decrescente, tem o IDX da feature
    rankfeature: str  # em ordem das features, tem o RANK de cada uma

    def __init__(self, *args, **kwargs):
        for k, i in kwargs.items():
            setattr(self, k, i)

    def __post_init__(self):
        pass


P = TypeVar("P")


# class ExtremeLearning(ELM, BaseEstimator):
#     """
#     Adaptor class to make compatible to Scikit API

#     https://hpelm.readthedocs.io/en/latest/api/elm.html

#     """

#     def __init__(self, *args, **kwargs):
#         #super().__init__(kwargs.get("inputs"), 2)
#         super().__init__(30, 2)  # [30, 2]
#         #self.add_neurons(
#         #    kwargs.get("neurons"), kwargs.get("activation"),
#         #)

#     def fit(self, x, y):
#         # return self.train(x, y, "CV", "OP", "c")  # needs number of folds
#         # return self.train(x, y, "V", "OP", "c")  # needs validation set
#         y = one_hot(y, 2)
#         return self.train(x, y, "LOO", "OP", "c")

#     def predict(self, x):
#         y = super().predict(x)
#         return one_hot_inverse(y)

#     def get_params(self, deep=True):
#         return {}

#     def set_params(self, *args, **kwargs):
#         #self.nnet.reset()
#         self.nnet.neurons = []
#         self.add_neurons(
#             kwargs.get("neurons"), kwargs.get("activation"),
#         )
#         return self

# class ExtremeLearning(ELM, BaseEstimator):
#     def __init__(self, hid_num=100, *args, **kwargs):
#         super().__init__(kwargs.get("hid_num"), 100)

#     def set_params(self, *args, **kwargs):
#         self.hid_num = kwargs.get("hid_num")
#         return self


def bayes_search_CV_init(self, estimator, search_spaces, optimizer_kwargs=None,
                         n_iter=50, scoring=None, fit_params=None, n_jobs=1,
                         n_points=1, iid=True, refit=True, cv=None, verbose=0,
                         pre_dispatch='2*n_jobs', random_state=None,
                         error_score='raise', return_train_score=False):

        self.search_spaces = search_spaces
        self.n_iter = n_iter
        self.n_points = n_points
        self.random_state = random_state
        self.optimizer_kwargs = optimizer_kwargs
        self._check_search_space(self.search_spaces)
        self.fit_params = fit_params

        super(BayesSearchCV, self).__init__(
             estimator=estimator, scoring=scoring,
             n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
             pre_dispatch=pre_dispatch, error_score=error_score,
             return_train_score=return_train_score)
        
BayesSearchCV.__init__ = bayes_search_CV_init


class ExtremeLearning(ELMClassifier):
    def _get_tags(self):
        return {
            "non_deterministic": False,
            "requires_positive_X": False,
            "requires_positive_y": False,
            #'X_types': ['2darray','sparse','categorical'],
            "X_types": ["2darray"],
            "poor_score": True,
            "no_validation": True,
            "multioutput": True,
            "allow_nan": False,
            "stateless": False,
            "multilabel": False,
            "_skip_test": False,
            "multioutput_only": False,
            "binary_only": False,
            "requires_fit": True,
            "pairwise": False,
        }


def humantime(*args, **kwargs):
    """
    Return time (duration) in human readable format.

    >>> humantime(seconds=3411)
    56 minutes, 51 seconds
    >>> humantime(seconds=800000)
    9 days, 6 hours, 13 minutes, 20 seconds
    """
    secs = float(timedelta(*args, **kwargs).total_seconds())
    units = [("day", 86400), ("hour", 3600), ("minute", 60), ("second", 1)]
    parts = []
    for unit, mul in units:
        if secs / mul >= 1 or mul == 1:
            if mul > 1:
                n = int(math.floor(secs / mul))
                secs -= n * mul
            else:
                # n = secs if secs != int(secs) else int(secs)
                n = int(secs) if secs != int(secs) else int(secs)
            parts.append("%s %s%s" % (n, unit, "" if n == 1 else "s"))
    return ", ".join(parts)


def loggerthread(q):
    """
    Main process thread receiver (handler) for log records.
    """
    while True:
        record = q.get()
        if record is None:
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)


def one_hot(array, num_classes):
    return np.squeeze(np.eye(num_classes)[array.reshape(-1)])


def one_hot_inverse(array):
    return np.argmax(array, axis=1)


def readdir(path) -> Dict[str, List[Tuple[np.ndarray, str]]]:
    """
    Read the CSV content of a directory into a list of numpy arrays.

    The return type is actually a dict with the "class" as key.
    """
    well_vars = [
        "P-PDG",
        "P-TPT",
        "T-TPT",
        "P-MON-CKP",
        "T-JUS-CKP",
        "P-JUS-CKGL",
        "T-JUS-CKGL",
        "QGL",
    ]
    columns = ["timestamp"] + well_vars + ["class"]
    r = []
    class_ = path[-1]
    with os.scandir(path) as it:
        for entry in it:
            if not entry.name.startswith(".") and entry.is_file():
                frame = pd.read_csv(entry, sep=",", header=0, names=columns)

                # str timestamp to float
                frame["timestamp"] = np.array(
                    [
                        pd.to_datetime(d).to_pydatetime().timestamp()
                        for d in frame.loc[:, "timestamp"]
                    ],
                    dtype=np.float64,
                )
                # cast int to float
                frame["class"] = frame["class"].astype(np.float64)

                # remember that scikit has problems with float64
                array = frame.loc[:, columns].to_numpy()

                r.append((array, entry.name))
    rd = {}
    rd[class_] = r
    return rd


def get_logging_config():
    return {
        "version": 1,
        "formatters": {
            "detailed": {
                "class": "logging.Formatter",
                "format": (
                    "%(asctime)s %(name)-12s %(levelname)-8s %(processName)-10s "
                    "%(module)-12s %(funcName)-15s %(message)s"
                ),
            }
        },
        "handlers": {
            "console": {"class": "logging.StreamHandler", "level": "INFO",},
            "file": {
                "class": "logging.FileHandler",
                "filename": "experiment1a.log",
                "mode": "w",
                "formatter": "detailed",
            },
            "errors": {
                "class": "logging.FileHandler",
                "filename": "experiment1a_errors.log",
                "mode": "w",
                "level": "ERROR",
                "formatter": "detailed",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file", "errors"]},
    }


def readdirparallel(path):
    """
    Read all CSV content of a directory in parallel.
    """
    njobs = psutil.cpu_count()
    results = []
    # with Pool(processes=njobs) as p:
    with ThreadPoolExecutor(max_workers=njobs) as p:
        # with ProcessPoolExecutor(max_workers=njobs) as p:
        # results = p.starmap(
        results = p.map(
            readdir, [os.path.join(path, str(c)) for c in [0, 1, 2, 3, 4, 5, 6, 7, 8]],
        )
    return results


def csv2bin(*args, **kwargs) -> None:
    """
    Read 3W dataset CSV files and save in a single numpy binary file.
    """
    raise Exception("not implemented")


def csv2hdf(*args, **kwargs) -> None:
    """
    Read 3W dataset CSV files and save in a single HDF5 file.
    """

    logger = logging.getLogger(f"clean")
    formatter = logging.Formatter(
        "%(asctime)s %(name)-12s %(levelname)-8s %(lineno)-5d %(funcName)-10s %(module)-10s %(message)s"
    )
    fh = logging.FileHandler(f"experiments_csv2hdf.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    path: str = kwargs.get("path")
    useclasses = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    well_vars = [
        "P-PDG",
        "P-TPT",
        "T-TPT",
        "P-MON-CKP",
        "T-JUS-CKP",
        "P-JUS-CKGL",
        "T-JUS-CKGL",
        "QGL",
    ]
    columns = ["timestamp"] + well_vars + ["class"]
    print("read CSV and save HDF5 ...", end="", flush=True)
    t0 = time.time()
    with h5py.File("datasets.h5", "w") as f:
        for c in useclasses:
            grp = f.create_group(f"/{c}")
            with os.scandir(os.path.join(path, str(c))) as it:
                for entry in it:
                    # if not entry.name.startswith(".") and entry.is_file():
                    if (
                        not entry.name.startswith(".")
                        and entry.is_file()
                        and "WELL" in entry.name
                    ):
                        frame = pd.read_csv(entry, sep=",", header=0, names=columns)

                        # str timestamp to float
                        frame["timestamp"] = np.array(
                            [
                                pd.to_datetime(d).to_pydatetime().timestamp()
                                for d in frame.loc[:, "timestamp"]
                            ],
                            dtype=np.float64,
                        )
                        # cast int to float
                        frame["class"] = frame["class"].astype(np.float64)

                        # remember that scikit has problems with float64
                        array = frame.loc[:, columns].to_numpy()

                        # entire dataset is float, incluinding timestamp & class labels
                        grp.create_dataset(
                            f"{entry.name}", data=array, dtype=np.float64
                        )
    print(f"finished in {time.time()-t0:.1}s.")


def csv2hdfpar(*args, **kwargs) -> None:
    """
    Read 3W dataset CSV files and save in a single HDF5 file.
    """
    path: str = kwargs.get("path")
    # useclasses = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    print("read CSV and save HDF5 ...", end="", flush=True)
    t0 = time.time()
    with h5py.File("datasets.h5", "w") as f:
        datalist = readdirparallel(path)
        for dd in datalist:
            for key in dd:
                grp = f.create_group(f"/{key}")
                for (array, name) in dd[key]:
                    # entire dataset is float, incluinding timestamp & class labels
                    grp.create_dataset(f"{name}", data=array, dtype=np.float64)
    print(
        f"finished {humanize.naturalsize(os.stat('datasets.h5').st_size)} "
        f"in {humantime(seconds=time.time()-t0)}."
    )


"""
def csv2hdfparmpi(*args, **kwargs) -> None:
    #Read 3W dataset CSV files and save in a single HDF5 file.
    path: str = kwargs.get("path")
    useclasses = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    well_vars = [
        "P-PDG",
        "P-TPT",
        "T-TPT",
        "P-MON-CKP",
        "T-JUS-CKP",
        "P-JUS-CKGL",
        "T-JUS-CKGL",
        "QGL",
    ]
    columns = ["timestamp"] + well_vars + ["class"]
    print("read CSV and save HDF5 ...", end="", flush=True)
    t0 = time.time()
    rank = MPI.COMM_WORLD.rank
    num_processes = MPI.COMM_WORLD.size
    with h5py.File("datasets.h5", "w", driver='mpio', comm=MPI.COMM_WORLD) as f:
        for c in useclasses:
            if c % num_processes == rank:
                grp = f.create_group(f"/{c}")
                with os.scandir(os.path.join(path, str(c))) as it:
                    for entry in it:
                        if not entry.name.startswith(".") and entry.is_file():
                            frame = pd.read_csv(entry, sep=",", header=0, names=columns)

                            # str timestamp to float
                            frame["timestamp"] = np.array(
                                [
                                    pd.to_datetime(d).to_pydatetime().timestamp()
                                    for d in frame.loc[:, "timestamp"]
                                ],
                                dtype=np.float64,
                            )
                            # cast int to float
                            frame["class"] = frame["class"].astype(np.float64)

                            # remember that scikit has problems with float64
                            array = frame.loc[:, columns].to_numpy()

                            # entire dataset is float, incluinding timestamp & class labels
                            dset = grp.create_dataset(
                                f"{entry.name}", data=array, dtype=np.float64
                            )
    print(f"finished in {time.time()-t0:.1}s.")
"""


def cleandataset(*args, **kwargs) -> None:
    """
    Read the the single file (with whole dataset), remove NaN and save 1 file per class.
    """
    well_vars = [
        "P-PDG",
        "P-TPT",
        "T-TPT",
        "P-MON-CKP",
        "T-JUS-CKP",
        "P-JUS-CKGL",
        "T-JUS-CKGL",
        "QGL",
    ]
    columns = ["timestamp"] + well_vars + ["class"]
    print("Reading dataset...")
    with h5py.File("datasets.h5", "r") as f:
        for c in range(0, 9):
            print(f"Processing class {c}")
            k = f"/{c}"
            soma = 0
            for s in f[k]:
                n = f[k][s].shape[0]
                soma = soma + n
                # print(k, s, n, soma)
            data = np.zeros([soma, 10], dtype=np.float64)
            i1 = 0
            # manual concatenation
            for s in f[k]:
                i2 = i1 + f[k][s].shape[0]
                data[i1:i2, :] = f[k][s][()]
                i1 = i2
            frame = pd.DataFrame(data=data, columns=columns)
            for col in ["P-PDG", "P-TPT", "T-TPT", "P-MON-CKP", "T-JUS-CKP"]:
                frame[col].fillna(method="ffill", axis=0, inplace=True)

            fp = np.memmap(
                f"datasets_clean_{c}.dat", dtype="float64", mode="w+", shape=frame.shape
            )
            fp[:, ...] = frame.to_numpy()
            del fp

    print("finished")


def cleandataseth5(*args, **kwargs) -> None:
    """
    Read the the single file (with whole dataset), remove NaN and save 1 file per class.
    """
    well_vars = [
        "P-PDG",
        "P-TPT",
        "T-TPT",
        "P-MON-CKP",
        "T-JUS-CKP",
        "P-JUS-CKGL",
        "T-JUS-CKGL",
        "QGL",
    ]
    columns = ["timestamp"] + well_vars + ["class"]

    logger = logging.getLogger(f"clean")
    formatter = logging.Formatter(
        "%(asctime)s %(name)-12s %(levelname)-8s %(lineno)-5d %(funcName)-10s %(module)-10s %(message)s"
    )
    fh = logging.FileHandler(f"experiments_clean.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    usecols = np.array([1, 2, 3, 4, 5], dtype=np.int)
    good = [columns[i] for i, _ in enumerate(columns) if i in usecols]
    print("Read hdf5 dataset and drop NaN...")
    with h5py.File("datasets.h5", "r") as f:
        logger.debug("reading input file")
        with h5py.File("datasets_clean.h5", "w") as fc:
            logger.debug("created output file")
            dropped = {}
            original = {}
            nonz = {}
            nonz2 = {}
            for c in range(0, 9):
                dropped[c] = 0
                original[c] = 0
                nonz[c] = [0,0,0,0,0]
                nonz2[c] = [0,0,0,0,0]
                grp = fc.create_group(f"/{c}")
                logger.debug(f"Processing class {c}")
                k = f"/{c}"
                for s in f[k]:
                    if s[0] != "W":
                        continue

                    logger.debug(f"class{c} {s}")

                    data = f[k][s][()]

                    mask = np.any(
                        np.isnan(data[:, usecols])
                        | (data[:, usecols] > np.finfo(np.float32).max)
                        | np.isinf(data[:, usecols])
                        | ~np.isfinite(data[:, usecols]),
                        axis=1,
                    )
                    nzvar = np.count_nonzero(
                        np.isnan(data[:, usecols])
                        | (data[:, usecols] > np.finfo(np.float32).max)
                        | np.isinf(data[:, usecols])
                        | ~np.isfinite(data[:, usecols]),
                        axis=0
                    )
                    nonz[c] = nonz[c] + nzvar

                    # from numpy to pandas
                    frame = pd.DataFrame(data=data, columns=columns)
                    f = frame[good].isna()

                    if np.sum(nzvar) > 0:
                        breakpoint()

                    b = frame.shape[0]
                    original[c] = original[c] + b
                    #logger.info(f"class {c} before dropping: {frame.shape[0]}")

                    # drop nan
                    frame.dropna(inplace=True, how="any", subset=good, axis=0)

                    a = frame.shape[0]
                    #logger.info(f"{c} {s} after dropping: {frame.shape[0]}")

                    dropped[c] = dropped[c] + b - a

                    # from pandas back to numpy
                    array = frame.to_numpy()

                    # drop using numpy only
                    # array = data[~mask]

                    """
                    try:
                        nc = len(usecols)
                        corr1 = np.zeros((nc, nc), dtype=np.float64)
                        pers1 = np.zeros((nc, nc), dtype=np.float64)
                        for aa, bb in itertools.combinations(usecols, 2):
                            a = aa - 1
                            b = bb - 1
                            corr1[a, b], pers1[a, b] = pearsonr(array[:, a], array[:, b])

                        logger.debug('Pearson R' + str(corr1))
                        logger.debug('p-value  ' + str(pers1))
                    except:
                        pass
                    """

                    # check_nan(frame, logger)
                    n = check_nan(array[:, [1, 2, 3, 4, 5]], logger)
                    if n > 0:
                        # breakpoint()
                        logger.info(f"{c} {s} dataset contains {n} NaN")

                    grp.create_dataset(f"{s}", data=array, dtype=np.float64)
            
            logger.info(str(original))
            logger.info(str(dropped))
            logger.info(pformat(nonz, indent=4))

    return None


def check_nan(array, logger) -> int:
    """
    Check array for inf, nan of null values.
    """
    logger.debug("*" * 50)
    logger.debug('array shape:' + str(array.shape))
    n = 0

    test = array[array > np.finfo(np.float32).max]
    logger.debug(f"test for numpy float32overflow {test.shape}")
    n = n + test.shape[0]

    # non finite include: np.inf, np.posinf, np.neginf, np.nan
    rows, cols = np.nonzero(~np.isfinite(array))
    test = array[rows, cols]
    logger.debug(f"test for numpy non finite {test.shape}")
    n = n + test.shape[0]
    if n > 0:
        logger.debug('  found: ' + str(array[rows[0], :]))

    test = array[np.isinf(array)]
    logger.debug(f"test for numpy inf {test.shape}")
    n = n + test.shape[0]

    test = array[np.isnan(array)]
    logger.debug(f"test for numpy NaN {test.shape}")
    n = n + test.shape[0]

    test = array[pd.isna(array)]
    logger.debug(f"test for pandas NA {test.shape}")
    n = n + test.shape[0]

    test = array[pd.isnull(array)]
    logger.debug(f"test for pandas isnull {test.shape}")
    n = n + test.shape[0]

    logger.debug("*" * 50)

    return n


def get_config_combination_list(settings, default=None) -> List:
    """
    Given a list of hyperparameters return all combinations of that.
    """
    keys = list(settings)
    r = []
    for values in itertools.product(*map(settings.get, keys)):
        d = dict(zip(keys, values))
        if default is not None:
            d.update(default)
        r.append(d)
    return r


def get_classifiers(clflist, n_jobs=1, default=False) -> Dict:
    """
    Classifiers and combinations of hyperparameters.
    """
    classifiers = OrderedDict()
    classifiers["ADA"] = {
        # "config": get_config_combination_list(
        "config": (
            {
                "n_estimators": [5, 25, 50, 75, 100, 250, 500],
                "algorithm": ["SAMME", "SAMME.R"],
            },
            {"random_state": None},
        ),
        "opt": {
            "n_estimators": skopt.space.Integer(1, 500),
            "algorithm": skopt.space.Categorical(["SAMME", "SAMME.R"]),
        },
        # default
        # "default": {"random_state": None, "n_estimators": 50, "algorithm": "SAMME.R"},
        # cfg0 after grid search 3A
        # "default": {'algorithm': 'SAMME.R', 'n_estimators': 100, 'random_state': 1},
        # cfg1 after grid search 3A
        # "default": {'algorithm': 'SAMME.R', 'n_estimators': 75, 'random_state': 1},
        # cfg2 after grid search 3A
        # "default": {'algorithm': 'SAMME', 'n_estimators': 500, 'random_state': 1},
        # cfg3 after grid search 3A
        # "default": {'algorithm': 'SAMME.R', 'n_estimators': 250, 'random_state': 1},
        # cfg4 after grid search 3A
        # "default": {'algorithm': 'SAMME', 'n_estimators': 500, 'random_state': 1},
        # =========================
        # cfg0 after grid search 3B
        "default": {"algorithm": "SAMME.R", "n_estimators": 500, "random_state": 1},
        # cfg1 after grid search 3B
        # "default": {'algorithm': 'SAMME.R', 'n_estimators': 500, 'random_state': 1},
        # cfg2 after grid search 3B
        # "default": {'algorithm': 'SAMME.R', 'n_estimators': 500, 'random_state': 1},
        # cfg3 after grid search 3B
        # "default": {'algorithm': 'SAMME.R', 'n_estimators': 500, 'random_state': 1},
        # cfg4 after grid search 3B
        # "default": {'algorithm': 'SAMME.R', 'n_estimators': 500, 'random_state': 1},
        "model": AdaBoostClassifier,
    }
    classifiers["DT"] = {
        "config": get_config_combination_list(
            {
                "criterion": ["gini", "entropy"],
                "splitter": ["best", "random"],
                "max_depth": [None, 5, 10, 50],
                "min_samples_split": [2, 5, 10],
            },
            {"random_state": None},
        ),
        "default": {"random_state": None},
        "model": DecisionTreeClassifier,
        "opt": {
            "max_depth": skopt.space.Integer(1, 500),
            "min_samples_split": skopt.space.Integer(2, 500),
            "criterion": skopt.space.Categorical(['gini', 'entropy']),
            "splitter": skopt.space.Categorical(["best", "random"]),
        },
    }
    classifiers["GBOOST"] = {
        "config": get_config_combination_list(
            {
                # "loss": ["deviance", "exponential"],
                "n_estimators": [50, 100, 250],
                "min_samples_split": [2, 5, 10],
                # "max_depth": [None, 5, 10, 50],
                "max_depth": [5, 10, 50],
            },
            {"random_state": None},
        ),
        "default": {"random_state": None},
        "model": GradientBoostingClassifier,
        "opt": {
            "n_estimators": skopt.space.Integer(1, 500),
            "min_samples_split": skopt.space.Integer(2, 500),
            "max_depth": skopt.space.Integer(1, 500),
        },
    }
    classifiers["1NN"] = {
        "config": [[], []],
        "default": {
            "n_neighbors": 1,
            # "weights": "uniform",
            "weights": "distance",
            "algorithm": "auto",
            "leaf_size": 30,
            "p": 2,
            "n_jobs": 1,
        },
        "model": KNeighborsClassifier,
        "opt": {},
    }
    classifiers["5NN"] = {
        "config": [],
        "default": {
            "n_neighbors": 5,
            # "weights": "uniform",
            "weights": "distance",
            "algorithm": "auto",
            "leaf_size": 30,
            "p": 2,
            "n_jobs": 1,
        },
        "model": KNeighborsClassifier,
        "opt": {},
    }
    classifiers["3NN"] = {
        "config": [],
        "default": {
            "n_neighbors": 3,
            # "weights": "uniform",
            "weights": "distance",
            "algorithm": "auto",
            "leaf_size": 30,
            "p": 2,
            "n_jobs": 1,
        },
        "model": KNeighborsClassifier,
        "opt": {},
    }
    classifiers["KNN"] = {
        # "config": get_config_combination_list(
        "config": (
            {
                "n_neighbors": [1, 3, 5, 7, 10, 15],
                # "weights": ["uniform", "distance"],
                # "algorithm": ["ball_tree"],
                # "leaf_size": [5, 25, 50],
                #'p': [1, 2, 3],
            },
            {"n_jobs": n_jobs}
            # não usa random state
        ),
        "default": {"n_jobs": n_jobs},
        # cfg0 after grid seach for 3A
        # "default": {'n_neighbors': 3},
        # "default": {'n_neighbors': 10},
        # "default": {'n_neighbors': 10},
        # "default": {'n_neighbors': 7},
        # "default": {'n_neighbors': 7},
        # cfg0 after grid seach for 3B
        # "default": {'n_neighbors': 3},
        # "default": {'n_neighbors': 1},
        # "default": {'n_neighbors': 5},
        # "default": {'n_neighbors': 1},
        # "default": {'n_neighbors': 5},
        "model": KNeighborsClassifier,
        "opt": {"n_neighbors": skopt.space.Integer(1, 50)},
    }
    classifiers["RF"] = {
        # "config": get_config_combination_list(
        "config": (
            {
                #'bootstrap': [True, False],
                # "bootstrap": [True],
                # "criterion": ["gini", "entropy"],
                # "criterion": ["gini"],
                #'max_features': ['auto', 'sqrt'],
                #'max_features': ['auto'],
                "max_features": ["auto", 1, 2, 4, 6, 8],
                # "max_features": ['auto'],
                # "max_depth": [None, 5, 10, 50],
                # "min_samples_split": [2, 4, 6, 8],
                # "min_samples_split": [2],
                #'min_samples_leaf': [1, 2, 5],
                # "min_samples_leaf": [1],
                "n_estimators": [100, 25, 50, 250],
                # "n_estimators": [100, 25],
            },
            # {"n_jobs": n_jobs, "random_state": None, "n_estimators": 100, "max_features": "auto"},
            {"n_jobs": n_jobs, "random_state": None,},
        ),
        # "default": {"random_state": None},
        "default": {"random_state": 1},
        # "best": {"random_state": None, "n_jobs": n_jobs, "n_estimators": 250, 'max_features': 2,},
        # "best": {'max_features': 2, 'n_estimators': 250, 'random_state': 1},
        # 3A best
        "best": {"max_features": 2, "n_estimators": 250, "random_state": 1},
        # 3B best
        # "best": {'max_features': 1, 'n_estimators': 100, 'random_state': 1},
        # 3A
        # "default": {'max_features': 2, 'n_estimators': 100, 'random_state': 1},
        # "default": {'max_features': 1, 'n_estimators': 25, 'random_state': 1},
        # "default": {'max_features': 1, 'n_estimators': 100, 'random_state': 1},
        # "default": {'max_features': 2, 'n_estimators': 250, 'random_state': 1},
        # "default": {'max_features': 2, 'n_estimators': 25, 'random_state': 1},
        # 3B
        # "default": {'max_features': 1, 'n_estimators': 100, 'random_state': 1},
        # "default": {'max_features': 4, 'n_estimators': 250, 'random_state': 1},
        # "default": {'max_features': 4, 'n_estimators': 250, 'random_state': 1},
        # "default": {'max_features': 1, 'n_estimators': 250, 'random_state': 1},
        # "default": {'max_features': 4, 'n_estimators': 100, 'random_state': 1},
        "model": RandomForestClassifier,
        "opt": {
            "max_features": skopt.space.Integer(1, 50),
            "n_estimators": skopt.space.Integer(1, 500),
        },
    }
    classifiers["SVM"] = {
        # "config": get_config_combination_list(
        "config": (
            {
                # "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
                # "kernel": ["linear", "poly", "rbf", "sigmoid"],
                # "gamma": ["scale", "auto"],
                "gamma": [0.001, 0.01, 0.1],
                "C": [0.001, 0.01, 0.1, 1.0],
                # "C": [1.0],
            },
            {},
        ),
        # default scikit
        "default": {"C": 1.0, "gamma": 0.01,},
        # best from grid search
        "best": {"C": 1.0, "gamma": 0.1},
        # cfg0 after grid search for 3A
        # "default": {'C': 1.0, 'gamma': 0.1},
        # cfg1 after grid search for 3A
        # "default": {'C': 1.0, 'gamma': 0.1},
        # cfg2 after grid search for 3A
        # "default": {'C': 1.0, 'gamma': 0.1},
        # cfg3 after grid search for 3A
        # "default": {'C': 1.0, 'gamma': 0.1},
        # cfg4 after grid search for 3A
        # "default": {'C': 1.0, 'gamma': 0.1},
        # cfg0 after grid search for 3B
        # "default": {'C': 1.0, 'gamma': 0.1},
        # cfg1 after grid search for 3B
        # "default": {'C': 1.0, 'gamma': 0.1},
        # cfg2 after grid search for 3B
        # "default": {'C': 1.0, 'gamma': 0.1},
        # cfg3 after grid search for 3B
        # "default": {'C': 1.0, 'gamma': 0.1},
        # cfg4 after grid search for 3B
        # "default": {'C': 1.0, 'gamma': 0.1},
        "model": SVC,
        "opt": {
            "gamma": skopt.space.Real(0.0001, 10),
            "C": skopt.space.Real(0.0001, 10),
        },
    }
    classifiers["GNB"] = {
        "config": [[], []],
        "default": {},
        "model": "GaussianNB",
        "opt": {},
    }
    classifiers["LDA"] = {
        "config": [[], []],
        "default": {},
        "model": LinearDiscriminantAnalysis,
        "opt": {},
    }
    classifiers["QDA"] = {
        "config": [[], []],
        "default": {},
        "model": QuadraticDiscriminantAnalysis,
        "opt": {},
    }
    classifiers["LGBM"] = {
        "config": get_config_combination_list(
            {
                "num_leaves": [50],
                "learning_rate": [0.1],
                "num_iterations": [100],
                "top_k": [20],
                # "min_data_per_group"
                # "min_gain_to_split"
                # "linear_lambda"
                # "min_data_in_leaf"
                "max_depth": [-1],
                "boosting": ["gbdt"],
            },
            {"device_type": "cpu", "random_state": None},
        ),
        "default": {"random_state": None},
        "model": LGBMClassifier,
        "opt": {},
    }
    classifiers["XGB"] = {
        "config": get_config_combination_list(
            {
                "n_estimators": [100],
                "max_depth": [50],
                "learning_rate": [0.1],
                "objective": ["binary:logistic"],
                "booster": ["gbtree", "gblinear", "dart"],
                "tree_method": ["auto"],
            }
        ),
        "default": {},
        "model": XGBClassifier,
        "opt": {},
    }
    classifiers["CATB"] = {
        "config": get_config_combination_list(
            {"iterations": [None], "learning_rate": [None], "depth": [None],}
        ),
        "default": {},
        "model": CatBoostClassifier,
        "opt": {},
    }
    classifiers["GAUSSNB"] = {
        "config": get_config_combination_list({}
        ),
        "default": None,
        "model": GaussianNB,
        "opt": {},
    }

    """
    classifiers["GAUSSNB"] = {
        "config": get_config_combination_list({}
        ),
        "default": None,
        "model": GaussianNB,
    }
    classifiers["EXTRAT1"] = {
        "config": get_config_combination_list({}
        ),
        "default": None,
        "model": ExtraTreeClassifier,
    }
    classifiers["EXTRAT2"] = {
        "config": get_config_combination_list({}
        ),
        "default": None,
        "model": ExtraTreesClassifier,
    }
    classifiers["BAG"] = {
        "config": get_config_combination_list({}
        ),
        "default": None,
        "model": BaggingClassifier,
    }
    classifiers["STACK"] = {
        "config": get_config_combination_list({}
        ),
        "default": None,
        "model": StackingClassifier,
    }
    classifiers["VOTE"] = {
        "config": get_config_combination_list({}
        ),
        "default": None,
        "model": VotingClassifier,
    }
    classifiers["HISTG"] = {
        "config": get_config_combination_list({}
        ),
        "default": None,
        "model": HistGradientBoostingClassifier,
    }
    classifiers["RADN"] = {
        "config": get_config_combination_list({}
        ),
        "default": None,
        "model": RadiusNeighborsClassifier,
    }
    classifiers["NEARC"] = {
        "config": get_config_combination_list({}
        ),
        "default": None,
        "model": NearestCentroid,
    }
    
    classifiers["DUMMY-STRAT"] = {
        "config": get_config_combination_list(
            {
                "strategy": [
                    "stratified",
                    # "most_frequent",
                    # "prior",
                    # "uniform",
                    # "constant"
                ],
            },
            {"random_state": None},
        ),
        "default": {"strategy": "stratified", "random_state": None},
        "model": DummyClassifier,
    }
    classifiers["DUMMY-UNI"] = {
        "config": get_config_combination_list(
            {
                "strategy": [
                    # "stratified",
                    # "most_frequent",
                    # "prior",
                    "uniform",
                    # "constant"
                ],
            },
            {"random_state": None},
        ),
        "default": {"strategy": "uniform", "random_state": None},
        "model": DummyClassifier,
    }
    classifiers["DUMMY-MF"] = {
        "config": get_config_combination_list(
            {
                "strategy": [
                    # "stratified",
                    "most_frequent",
                    # "prior",
                    # "uniform",
                    # "constant"
                ],
            },
            {"random_state": None},
        ),
        "default": {"strategy": "most_frequent", "random_state": None},
        "model": DummyClassifier,
    }
    """

    classifiers["MLP"] = {
        "config": (
            {
                # "hidden_layer_sizes": [64, 128, 256, 512],
                "hidden_layer_sizes": [64, 100, 256, 512],
                # "solver": ["lbfgs", "sgd", "adam"],
                # "activation": ["identity", "logistic", "tanh", "relu"],
                # "activation": ["relu", "logistic"],
                "activation": ["relu"],
                # "max_iter": [500],
                "max_iter": [100, 200, 500],
                # "shuffle": [True],
                # "momentum": [0.9],
                # "power_t": [0.5],
                # "learning_rate": ["constant", "invscaling", "adaptive"],
                # "learning_rate": ["constant"],
                # "batch_size": ["auto"],
                # "alpha": [0.0001],
            },
            {"random_state": None},
        ),
        # "default": {"random_state": None, "max_iter": 200, "hidden_layer_sizes": 100,},
        # cfg0 after grid search for 3A
        # "default": {'activation': 'relu', 'hidden_layer_sizes': 64, 'max_iter': 500},
        # cfg1 after grid search for 3A
        # "default": {'activation': 'relu', 'hidden_layer_sizes': 256, 'max_iter': 200},
        # cfg2 after grid search for 3A
        # "default": {'activation': 'relu', 'hidden_layer_sizes': 512, 'max_iter': 200},
        # cfg3 after grid search for 3A
        # "default": {'activation': 'relu', 'hidden_layer_sizes': 100, 'max_iter': 200},
        # cfg4 after grid search for 3A
        # "default": {'activation': 'relu', 'hidden_layer_sizes': 64, 'max_iter': 200},
        # cfg0 after grid search for 3B
        # "default": {'activation': 'relu', 'hidden_layer_sizes': 512, 'max_iter': 200},
        # cfg1 after grid search for 3B
        # "default": {'activation': 'relu', 'hidden_layer_sizes': 512, 'max_iter': 500},
        # cfg2 after grid search for 3B
        # "default": {'activation': 'relu', 'hidden_layer_sizes': 512, 'max_iter': 500},
        # cfg3 after grid search for 3B
        # "default": {'activation': 'relu', 'hidden_layer_sizes': 256, 'max_iter': 500},
        # cfg4 after grid search for 3B
        "default": {"activation": "relu", "hidden_layer_sizes": 256, "max_iter": 200},
        "model": MLPClassifier,
        "opt": {},
    }
    classifiers["ZERORULE"] = {
        "config": (
            {
                "strategy": [
                    # "stratified",
                    # "most_frequent",
                    # "prior",
                    # "uniform",
                    # "constant"
                    "most_frequent"
                ],
            },
            {"random_state": None},
        ),
        # "default": {"strategy": "constant"},
        "default": {"strategy": "most_frequent", "random_state": None},
        "model": DummyClassifier,
        "opt": {},
    }

    # https://hpelm.readthedocs.io/en/latest/api/elm.html
    classifiers["ELM"] = {
        # "config": get_config_combination_list(
        "config": (
            {
                # "neurons": [50, 100, 250, 500, 1000, 2500],
                # "hid_num": [50, 100, 250, 500, 1000, 2500],
                "n_neurons": [None, 50, 100, 250, 500, 1000, 2500],
                "ufunc": ["tanh", "sigmoid"],
                # "activation": [
                #    "tanh"
                # ],  # “lin” for linear, “sigm” or “tanh” for non-linear, “rbf_l1”, “rbf_l2” or “rbf_linf”
            },
            {"random_state": None},
        ),
        # "default": {"neurons": 100, "activation": "tanh"},
        # "default": {"hid_num": 100},
        # "default": {"n_neurons": None, "random_state": None},
        # cfg0 after grid search for 3A
        # "default": {'n_neurons': None, 'random_state': 1, 'ufunc': 'tanh'},
        # cfg1 after grid search for 3A
        # "default": {'n_neurons': 1000, 'random_state': 1, 'ufunc': 'tanh'},
        # cfg2 after grid search for 3A
        # "default": {'n_neurons': 1000, 'random_state': 1, 'ufunc': 'tanh'},
        # cfg3 after grid search for 3A
        # "default": {'n_neurons': 1000, 'random_state': 1, 'ufunc': 'tanh'},
        # cfg4 after grid search for 3A
        # "default": {'n_neurons': 500, 'random_state': 1, 'ufunc': 'tanh'},
        # cfg0 after grid search for 3B
        # "default": {'n_neurons': 1000, 'random_state': 1, 'ufunc': 'tanh'},
        # cfg1 after grid search for 3B
        # "default": {'n_neurons': 1000, 'random_state': 1, 'ufunc': 'tanh'},
        # cfg2 after grid search for 3B
        "default": {"n_neurons": 500, "random_state": 1, "ufunc": "tanh"},
        # cfg3 after grid search for 3B
        # "default": {'n_neurons': 1000, 'random_state': 1, 'ufunc': 'tanh'},
        # cfg4 after grid search for 3B
        # "default": {'n_neurons': 1000, 'random_state': 1, 'ufunc': 'tanh'},
        "model": ExtremeLearning,
        # "model": ELMClassifier,
        # "model": ELM,
        "opt": {},
    }

    if isinstance(clflist, str):
        if not clflist or clflist.lower() != "all":
            clflist = clflist.split(",")
    elif isinstance(clflist, tuple):
        clflist = list(clflist)

    if default:
        for c in classifiers.keys():
            """
            if c[:5] != "DUMMY":
                classifiers[c]['config'] = {}
            else:
                del classifiers[c]
            """
            classifiers[c]["config"] = {}
        return classifiers
    else:
        ret = {}
        for c in clflist:
            if c in classifiers.keys():
                ret[c] = classifiers[c]
        return ret


def _split_data(n: int, folds: int) -> Sequence[Tuple[int, int]]:
    """
    Return list of tuples with array index for each fold.
    """
    raise Exception("depends on which experiment")


def _read_and_split_h5(fold: int, params: P) -> Tuple:
    """
    HDF files offer at least a couple advantages:
    1 - reading is faster than CSV
    2 - you dont have to read the whole dataset to get its size (shape)

    H5PY fancy indexing is very slow.
    https://github.com/h5py/h5py/issues/413
    """
    raise Exception("depends on which experiment")


def _read_and_split_bin(fold: int, params: P) -> Tuple:
    """
    HDF files offer at least a couple advantages:
    1 - reading is faster than CSV
    2 - you dont have to read the whole dataset to get its size (shape)

    Numpy needs to know 'shape' beforehand.
    https://numpy.org/doc/stable/reference/generated/numpy.memmap.html

    """
    raise Exception("depends on which experiment")


def train_test_binary(
    classif, xtrain, ytrain, xtest, ytest
) -> Tuple[Tuple, Tuple, Tuple, Tuple]:
    """
    Execute training and testing (binary) and return metrics (F1, Accuracy, CM)
    """
    p = 0.0
    r = 0.0
    f1binp, f1binn = 0.0, 0.0
    f1mic = 0.0
    f1mac = 0.0
    f1weigh = 0.0
    f1sam = 0.0
    acc = 0.0
    excp = []
    excptb = []
    ypred = np.full([xtrain.shape[0]], 0, dtype="int")
    ynpred = 0
    yppred = 0
    tn = 0
    tp = 0
    fp = 0
    fn = 0
    trt1 = 0.0
    try:

        trt0 = time.time()
        classif.fit(xtrain, ytrain)
        trt1 = time.time() - trt0

        try:
            ypred = classif.predict(xtest)
            yppred = np.sum(ypred)
            ynpred = ytest.shape[0] - yppred

            try:
                p, r, f1binp, _ = precision_recall_fscore_support(
                    ytest, ypred, average="binary", pos_label=1,
                )
                _, _, f1binn, _ = precision_recall_fscore_support(
                    ytest, ypred, average="binary", pos_label=0,
                )
            except Exception as ef1bin:
                # f1_score(ytest, ypred, average="binary")
                excp.append(ef1bin)
                # raise ef1bin

            try:
                f1mic = f1_score(ytest, ypred, average="micro")
            except Exception as e2:
                excp.append(e2)

            try:
                f1mac = f1_score(ytest, ypred, average="macro")
            except Exception as e3:
                excp.append(e3)

            try:
                f1weigh = f1_score(ytest, ypred, average="weighted")
            except Exception as e4:
                excp.append(e4)

            try:
                acc = accuracy_score(ytest, ypred)
            except Exception as e_acc:
                excp.append(e_acc)
                # raise e_acc

            try:
                tn, fp, fn, tp = confusion_matrix(
                    ytest, ypred, labels=[0, 1], sample_weight=None, normalize=None,
                ).ravel()
            except Exception as ecm:
                excp.append(ecm)
                # raise ecm

        except Exception as e_pred:
            excp.append(e_pred)
            raise e_pred

    except Exception as efit:
        excp.append(efit)
        # exc_type, exc_value, exc_traceback = sys.exc_info()
        einfo = sys.exc_info()
        # excptb.append(exc_traceback)
        excptb.append(einfo[2])
        # breakpoint()
        raise efit

    return (
        (ypred, ynpred, yppred, trt1),
        (f1binp, f1binn, f1mic, f1mac, f1weigh, f1sam, p, r, acc),
        (tn, fp, fn, tp),
        tuple(excp),
    )


def vertical_split_bin(negative, positive):
    x = np.concatenate([negative, positive], axis=0).astype(np.float32)
    y = np.concatenate(
        [
            np.zeros(negative.shape[0], dtype=np.int32),
            np.ones(positive.shape[0], dtype=np.int32),
        ],
        axis=0,
    )
    return x, y


def horizontal_split_file(fold, nfolds, files, seed):
    count = 0
    samples = []
    sessions = []
    for s in files:
        if s[0] != "W":
            continue
        n = files[s].shape[0]
        count += 1
        samples.append(n)
        sessions.append(s)

    samples = np.array(samples)
    nf = np.int(count / nfolds)

    # sequence
    # testfidx = np.reshape(np.arange(nf * nfolds), (-1, 5))
    testfidx = np.reshape(np.arange(nf * nfolds), (5, -1)).T

    # random
    # testfidx = np.random.RandomState(seed).choice(
    #    range(0, count),
    #    size=(nf, nfolds),
    #    replace=False,
    # )
    # print('files:', testfidx)

    testsize = sum(samples[testfidx[:, fold]])
    test = np.zeros((testsize, 10), dtype=np.float64)
    # stest = sessions[testfidx]
    stest = [sessions[i] for i in testfidx[:, fold]]

    i1 = 0
    i2 = 0
    for s in stest:
        i2 = i1 + files[s].shape[0]
        test[i1:i2, :] = files[s][()]
        i1 = i2
        # print(s)
    print(f"fold {fold} ate {i2}")
    # breakpoint()
    # nstp = sum(splits[f][2] for f in range(nfolds) if f != fold)
    nstp = 0
    for s in sessions:
        if s in stest:
            continue
        nstp += files[s].shape[0]

    train = np.zeros((nstp, 10), dtype=np.float64)
    i1 = 0
    i2 = 0
    for k, s in enumerate(sessions):
        if s in stest:
            continue
        n = files[s].shape[0]
        i2 = i1 + n
        train[i1:i2, :] = files[s][()]
        i1 = i2
        # print(k, s, n, nstp)
    # breakpoint()

    return train, test


def horizontal_split_well(fold, nfolds, file, seed=None):
    wellstest = [1, 2, 4, 5, 7]
    welltest = wellstest[fold]
    count = 0
    wells = {}
    stest = []
    for s in file:
        if s[0] != "W":
            continue
        n = file[s].shape[0]
        count += 1
        welli = int(str(s[6:10]))
        if welli not in wells:
            wells[welli] = 0
        wells[welli] += n
        if welli == welltest:
            stest.append(s)
    # print(wells)
    # print(f'fold {fold}', wells.keys())
    if wellstest[fold] in wells:
        ntest = wells[wellstest[fold]]
        test = np.zeros((ntest, 10), dtype=np.float64)
        i1 = 0
        i2 = 0
        for s in stest:
            if s[0] != "W":
                continue
            i2 = i1 + file[s].shape[0]
            test[i1:i2, :] = file[s][()]
            i1 = i2
    else:
        print("data for this fault and well not available")
        test = np.empty((0, 10), dtype=np.float64)

    ntrain = sum(wells[k] for k in wells if k != welltest)
    train = np.zeros((ntrain, 10), dtype=np.float64)
    i1 = 0
    i2 = 0
    for s in file:
        if s[0] != "W":
            continue
        if s in stest:
            continue
        i2 = i1 + file[s].shape[0]
        train[i1:i2, :] = file[s][()]
        i1 = i2
        # print('well', s, i1, i2, ntrain)

    return train, test


def drop_nan(*args):
    for a in args:
        mask = np.any(
            np.isnan(a)
            # | (trainnegative > np.finfo(np.float32).max)
            | np.isinf(a) | ~np.isfinite(a),
            axis=1,
        )
        a = a[~mask]
    return args


def get_mask(*args):
    m = []
    for a in args:
        mask = np.any(
            np.isnan(a)
            # | (trainnegative > np.finfo(np.float32).max)
            | np.isinf(a) | ~np.isfinite(a),
            axis=1,
        )
        m.append(mask)
    return m


def savearray(filename, array):
    with open(filename, "wb") as f:
        np.save(f, array)


def loadarray(filename):
    pass


def save_datasets(xdata, ydata, round_, fold, case, scenario):
    pass


def get_md5(params):
    import hashlib

    experiment = f"nr{params.nrounds}_nf{params.nfolds}_w{params.windowsize}_s{params.stepsize}".encode(
        "utf-8"
    )
    return hashlib.md5(experiment).hexdigest()


def split_and_save1(params, case, group, classes):
    """
    """
    win = params.windowsize
    step = params.stepsize
    # filename = get_md5(params)

    with h5py.File(f"datasets_clean.h5", "r") as file:
        n = 0
        skipped = 0
        for c in classes:
            f = file[f"/{c}"]
            for s in f:
                if s[0] != "W":
                    continue
                if len(params.skipwell) > 0:
                    # skip well by ID
                    if s[:10] in params.skipwell:
                        skipped += f[s].shape[0]
                        continue
                n += f[s].shape[0]

        data = np.zeros([n, 10], dtype=np.float64)
        test = np.zeros((skipped, 10), dtype=np.float64)

        for c in classes:
            f = file[f"/{c}"]
            for s in f:
                i1, i2 = 0, 0
                j1, j2 = 0, 0
                for s in f:
                    if s[0] != "W":
                        continue
                    if len(params.skipwell) > 0:
                        if s[:10] in params.skipwell:
                            j2 = j1 + f[s].shape[0]
                            test[j1:j2, :] = f[s][()]
                            j1 = j2
                            continue
                    i2 = i1 + f[s].shape[0]
                    data[i1:i2, :] = f[s][()]
                    i1 = i2

        # NaN - not a number
        # m = get_mask(data[:, usecols])
        # data = data[~m]

        xdata = swfe(params.windowsize, n, params.stepsize, data[:, params.usecols],)

        tdata = swfe(
            params.windowsize, skipped, params.stepsize, test[:, params.usecols],
        )

        if group == "pos":
            ydata = np.ones(xdata.shape[0], dtype=np.float64)
        elif group == "neg":
            ydata = np.zeros(xdata.shape[0], dtype=np.float64)

        with h5py.File(f"datasets_folds_exp{case}.h5", "a") as ffolds:
            for round_ in range(1, params.nrounds + 1):
                if params.shuffle:
                    kf = KFold(
                        n_splits=params.nfolds, random_state=round_, shuffle=True
                    )
                else:
                    kf = KFold(n_splits=params.nfolds, random_state=None, shuffle=False)
                for fold, (train_index, test_index) in enumerate(kf.split(xdata)):
                    gk = f"/case{case}_{group}_r{round_}_nf{params.nfolds}_f{fold}_w{win}_s{step}"

                    if gk in ffolds:
                        del ffolds[gk]

                    grp = ffolds.create_group(gk)

                    xtrain, ytrain = xdata[train_index], ydata[train_index]
                    xtest, ytest = xdata[test_index], ydata[test_index]

                    print(
                        gk,
                        "original data shape",
                        data.shape,
                        "final",
                        xdata.shape,
                        "xtrain",
                        xtrain.shape,
                        "xtest",
                        xtest.shape,
                    )

                    grp.create_dataset(f"xtrain", data=xtrain, dtype=np.float64)
                    grp.create_dataset(f"ytrain", data=ytrain, dtype=np.float64)
                    grp.create_dataset(f"xvalid", data=xtest, dtype=np.float64)
                    grp.create_dataset(f"yvalid", data=ytest, dtype=np.float64)

                    if tdata.shape[0] > 0:
                        gkt = f"/case{case}_{group}_r{round_}_nf{params.nfolds}_f-test_w{win}_s{step}"
                        if gkt in ffolds:
                            del ffolds[gkt]
                        grpt = ffolds.create_group(gkt)
                        grpt.create_dataset(f"xtest", data=tdata, dtype=np.float64)


def split_and_save2(params, case, group, classes):
    win = params.windowsize
    step = params.stepsize
    nfolds = params.nfolds
    filename = get_md5(params)
    with h5py.File(f"datasets_clean.h5", "r") as file:
        with h5py.File(f"datasets_folds_exp{case}.h5", "a") as ffolds:
            for round_ in range(1, params.nrounds + 1):
                samples = []
                sessions = []
                for class_ in classes:
                    files = file[f"/{class_}"]
                    for s in files:
                        if s[0] != "W":
                            continue
                        n = files[s].shape[0]
                        samples.append(n)
                        sessions.append(f"/{class_}/{s}")
                count = len(samples)
                samples = np.array(samples)
                nf = np.int(count / nfolds)

                # random
                if params.shuffle:
                    testfidx = np.random.RandomState(round_).choice(
                        range(0, count), size=(nf, params.nfolds), replace=False,
                    )
                else:
                    # sequence
                    testfidx = np.reshape(np.arange(nf * nfolds), (5, -1)).T

                for fold in range(0, params.nfolds):
                    gk = f"/case{case}_{group}_r{round_}_nf{params.nfolds}_f{fold}_w{win}_s{step}"
                    testsize = sum(samples[testfidx[:, fold]])
                    test = np.zeros((testsize, 10), dtype=np.float64)
                    stest = [sessions[i] for i in testfidx[:, fold]]

                    i1, i2 = 0, 0
                    # for class_ in classes:
                    #    files = file[f"/{class_}"]
                    for s in stest:
                        i2 = i1 + file[s].shape[0]
                        test[i1:i2, :] = file[s][()]
                        i1 = i2
                        # print(s)
                    # print(f'fold {fold} ate {i2}')

                    nstp = 0
                    for s in sessions:
                        if s in stest:
                            continue
                        nstp += file[s].shape[0]

                    train = np.zeros((nstp, 10), dtype=np.float64)
                    i1, i2 = 0, 0
                    for s in sessions:
                        if s in stest:
                            continue
                        i2 = i1 + file[s].shape[0]
                        train[i1:i2, :] = file[s][()]
                        i1 = i2

                    xtrain = swfe(
                        params.windowsize,
                        nstp,
                        params.stepsize,
                        train[:, params.usecols],
                    )
                    if classes == params.negative:
                        ytrain = np.zeros(xtrain.shape[0], dtype=np.float64)
                    else:
                        ytrain = np.ones(xtrain.shape[0], dtype=np.float64)

                    xtest = swfe(
                        params.windowsize,
                        testsize,
                        params.stepsize,
                        test[:, params.usecols],
                    )

                    if classes == params.negative:
                        ytest = np.zeros(xtest.shape[0], dtype=np.float64)
                    else:
                        ytest = np.ones(xtest.shape[0], dtype=np.float64)

                    if gk in ffolds:
                        del ffolds[gk]

                    grp = ffolds.create_group(gk)

                    print(
                        gk,
                        "original data shape",
                        np.sum(samples),
                        "train",
                        train.shape,
                        "test",
                        test.shape,
                        "xtrain",
                        xtrain.shape,
                        "xtest",
                        xtest.shape,
                    )

                    grp.create_dataset(f"xtrain", data=xtrain, dtype=np.float64)
                    grp.create_dataset(f"ytrain", data=ytrain, dtype=np.float64)
                    grp.create_dataset(f"xvalid", data=xtest, dtype=np.float64)
                    grp.create_dataset(f"yvalid", data=ytest, dtype=np.float64)


def split_and_save3(params, case, group, classes):
    win = params.windowsize
    step = params.stepsize
    wellstest = [1, 2, 4, 5, 7]
    #filename = get_md5(params)
    with h5py.File(f"datasets_clean.h5", "r") as clean:
        print(f'datasets_clean.h5 => datasets_folds_exp{case}.h5')
        with h5py.File(f"datasets_folds_exp{case}.h5", "a") as ffolds:
            for round_ in range(1, params.nrounds + 1):
                for fold in range(0, params.nfolds):
                    print(f'{case} {round_} fold {fold} test against well {wellstest[fold]}')
                    gk = f"/case{case}_{group}_r{round_}_nf{params.nfolds}_f{fold}_w{win}_s{step}"
                    welltest = wellstest[fold]
                    count = 0
                    wells = {}
                    strain = []
                    stest = []
                    n = 0
                    for class_ in classes:
                        files = clean[f"/{class_}"]
                        for s in files:
                            # print(s[0])
                            if s[0] != "W":
                                continue
                            # n = files[s].shape[0]
                            count += 1
                            welli = int(str(s[6:10]))
                            if welli not in wells:
                                wells[welli] = 0
                            wells[welli] += files[s].shape[0]
                            if welli == welltest:
                                stest.append(f"/{class_}/{s}")
                            else:
                                strain.append(f"/{class_}/{s}")

                    ntrain = sum(wells[k] for k in wells if k != welltest)
                    train = np.zeros((ntrain, 10), dtype=np.float64)

                    if wellstest[fold] in wells:
                        ntest = wells[wellstest[fold]]
                        test = np.zeros((ntest, 10), dtype=np.float64)
                        i1, i2 = 0, 0
                        # for class_ in classes:
                        #    files = file[f'/{class_}']
                        for s in stest:
                            # if s[0] != "W":
                            #    continue
                            i2 = i1 + clean[s].shape[0]
                            test[i1:i2, :] = clean[s][()]
                            i1 = i2
                    else:
                        print("data for this fault and well not available")
                        test = np.empty((0, 10), dtype=np.float64)

                    # for class_ in classes:
                    #     files = clean[f"/{class_}"]
                    #     i1, i2 = 0, 0
                    #     for s in files:
                    #         s2 = f"/{class_}/{s}"
                    #         if s[0] != "W":
                    #             continue
                    #         if s2 in stest:
                    #             continue
                    #         i2 = i1 + files[s].shape[0]
                    #         train[i1:i2, :] = files[s][()]
                    #         i1 = i2
                    #         # print('well', s, i1, i2, ntrain)
                    i1, i2 = 0, 0
                    for s in strain:
                        i2 = i1 + clean[s].shape[0]
                        train[i1:i2, :] = clean[s][()]
                        i1 = i2

                    xtrain = swfe(
                        params.windowsize,
                        ntrain,
                        params.stepsize,
                        train[:, params.usecols],
                    )
                    if classes == params.negative:
                        ytrain = np.zeros(xtrain.shape[0], dtype=np.float64)
                    else:
                        ytrain = np.ones(xtrain.shape[0], dtype=np.float64)

                    xtest = swfe(
                        params.windowsize,
                        ntest,
                        params.stepsize,
                        test[:, params.usecols],
                    )

                    if classes == params.negative:
                        ytest = np.zeros(xtest.shape[0], dtype=np.float64)
                    else:
                        ytest = np.ones(xtest.shape[0], dtype=np.float64)

                    if params.shuffle:
                        # xtrain, ytrain, xtest, ytest = shuffle(xtrain, ytrain, xtest, ytest, random_state=round_)
                        xtrain, ytrain = resample(
                            xtrain, ytrain, random_state=round_, replace=False
                        )
                        xtest, ytest = resample(
                            xtest, ytest, random_state=round_, replace=False
                        )

                    if gk in ffolds:
                        del ffolds[gk]

                    grp = ffolds.create_group(gk)

                    print(gk, "xtrain", xtrain.shape, "xtest", xtest.shape)

                    grp.create_dataset(f"xtrain", data=xtrain, dtype=np.float64)
                    grp.create_dataset(f"ytrain", data=ytrain, dtype=np.float64)
                    grp.create_dataset(f"xvalid", data=xtest, dtype=np.float64)
                    grp.create_dataset(f"yvalid", data=ytest, dtype=np.float64)


def fs5foldcv(round_: int, fold: int, params: P, *args, **kwargs) -> None:
    """
    Cross-validation for feature selection
    """
    logging.captureWarnings(True)
    logger = logging.getLogger(f"fold{fold}")
    formatter = logging.Formatter(params.logformat)
    fh = logging.FileHandler(f"{params.experiment}_feature_cv_{params.fsdirection}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    logger.debug(f"round {round_}")
    #logger.debug(f"fold {fold}")

    #classifiers = get_classifiers(params.classifierstr)

    # ckf = CustomKFold(params=params, random_state=1)
    ckf = WellKFold(params=params, random_state=1)
    clf = "RF"
    cfg = 'default'


    try:
        classifiers = get_classifiers([clf])
        classifier = get_classifiers([clf]).get(clf)
        #estimator = classifier['model'](**classifier['default'])
        #direction = 'backward'
        direction = params.fsdirection
        logger.debug(f'sequential feature selection {direction}')
        resultdata = []
        adata = []



        def ranking_svm(x, y):
            """
            univariate feature estimate
            """
            n = 5
            rank = np.zeros(n, dtype=float)
            for it in range(n):
                rank[it] = evaluate([it], x, y, cfg="default", clf="SVM")
            return list(np.argsort(rank)[::-1]), rank

        def ranking_rf(x, y):
            n = 5
            rank = np.zeros(n, dtype=float)
            for it in range(n):
                rank[it] = evaluate([it], x, y, cfg="default", clf="RF")
            return list(np.argsort(rank)[::-1]), rank

        def ranking_anova(x, y):
            fsf = SelectKBest(f_classif, k="all")
            fsf.fit(x, y)
            return list(np.argsort(fsf.scores_)[::-1]), fsf.scores_

        def ranking_mi(x, y):
            # mi = SelectKBest(chi2, k='all')
            # mi.fit(x_data, y_data)
            miarray = mutual_info_classif(
                x, y, discrete_features=False, n_neighbors=3, copy=True, random_state=1
            )
            # return list(np.argsort(mi.scores_)[::-1]), fsc.scores_
            return list(np.argsort(miarray)[::-1]), miarray

        def ranking_lasso(x, y):
            rank = np.zeros(n, dtype=float)
            coef = np.zeros((5, 55), dtype=float)
            for f in range(5):
                for dk, data in enumerate(ckf.data_splits()):
                    xtrain, ytrain, xtest, ytest = data
                    estimator = Lasso()
                    estimator.fit(xtrain, ytrain)
                    # ypred = estimator.predict(xtest[:, [it]])
                    # try:
                    #    f1mac = f1_score(ytest, ypred, average="macro")
                    # except Exception as e3:
                    #    excp.append(e3)
                    # tmpscores[dk] = f1mac
                    coef[f, :] = np.abs(estimator.coef_)

            rank = np.mean(coef, axis=0)
            return list(np.argsort(rank)[::-1]), rank

        def evaluate(subset, x, y, clf='RF', cfg="default"):
            classifier = classifiers[clf]
            estimator = classifier["model"](**classifier[cfg])
            scores = cross_validate(estimator, x[:, subset], y, cv=5, n_jobs=5,scoring=["f1_macro", "accuracy"],
                error_score="raise",)
            return np.mean(scores["test_f1_macro"])
        
        def evaluate1(s, cfg="default", clf="RF"):
            tmpscores = np.zeros(5, dtype=float)
            classifier = classifiers[clf]
            a = []
            b = []
            c = []
            d = []
            e = []
            f = []
            for dk, data in enumerate(ckf.data_splits()):
                xtrain, ytrain, xtest, ytest = data
                a.append(xtrain[:, s])
                b.append(ytrain)
                c.append(xtest[:, s])
                d.append(ytest)
                e.append(clf)
                f.append(cfg)

            with Pool(processes=params.njobs) as p:
                # 'results' is a List of Dict
                results = p.starmap(cv_fold, zip(e, f, a, b, c, d, [params] * 5))
                tmpscores = np.array(results)

            return np.mean(tmpscores)

        def evaluate2(s, cfg="default"):
            classifier = classifiers["RF"]
            estimator = classifier["model"](**classifier[cfg])
            scores = cross_validate(
                estimator,
                ckf.xdata[:, s],
                ckf.ydata,
                cv=ckf,
                n_jobs=5,
                scoring=["f1_macro", "accuracy"],
                error_score="raise",
            )
            return np.mean(scores["test_f1_macro"])

        def evaluate3(subset, x, y, clf='RF', cfg="default"):
            classifier = classifiers[clf]
            estimator = classifier["model"](**classifier[cfg])
            scores = cross_validate(estimator, x[:, subset], y, cv=5, n_jobs=5, scoring=["f1_macro", "accuracy"],
                error_score="raise",)
            return scores
        
        def evaluate4(subset, x, y, clf='RF', cfg="default", cv=None):
            classifier = classifiers[clf]
            estimator = classifier["model"](**classifier[cfg])
            logger.debug(f'inner cross-validation ')
            scores = cross_validate(
                estimator, 
                x[:, subset], 
                y, 
                cv=cv, 
                n_jobs=1, 
                scoring=["f1_macro", "accuracy"],
                error_score="raise",
            )
            return scores


        # read from disk into memory for faster access and reuse
        for kf, data in enumerate(ckf.split()):
            xtr, ytr, xte, yte = data
            adata.append((xtr, ytr, xte, yte))

        kfoldobj = KFold(n_splits=5, shuffle=True, random_state=round_)

        # SFS e SBS
        if direction == 'forward':
            myrange = range(1, 55, 1)
        elif direction == 'backward':
            myrange = range(54, 0, -1)
        
        # final results:
        if direction == "forward" and params.experiment == "experiment3a":
            myrange = [20,15,10,8,23]
        elif direction == "forward" and params.experiment == "experiment3b":
            myrange = [11, 12, 3, 11, 4]
        elif direction == "backward" and params.experiment == "experiment3a":
            myrange = [7, 9, 3, 4, 5]
        elif direction == "backward" and params.experiment == "experiment3b":
            myrange = [6, 4, 4, 6, 5]

        kfinner = InnerWellFold(outerfold=kf, params=params)
        for i in myrange:
            #tmp = {}
            #tmp['nof'] = i
            logger.debug(f'=== try selecting {i} features - {params.experiment} {direction}')

            for kf, data in enumerate(adata):
                # final
                if kf != myrange.index(i):
                    logger.info('skip fold - unnecessary')
                    continue
                #
                logger.debug(f'outer fold {kf}')
                tmp = {}
                tmp['nof'] = i

                xtr, ytr, xte, yte = data

                # =====================================================================
                # SFS ou SBS
                # =====================================================================
                sfs = None
                classif = None
                estimator = None
                estimator = classifier['model'](**classifier['default'])

                #kfoldinner = InnerWellFold(outerfold=kf, params=params)
                kfoldinner = InnerWellFold2(outerfold=kf)
                kfoldinner.xdata = kfinner.xdata
                kfoldinner.ydata = kfinner.ydata
                kfoldinner.splits = kfinner.splits

                #sfs = SequentialFeatureSelector(estimator, n_features_to_select=i, direction=direction, cv=ckf, n_jobs=5, scoring='f1_macro')
                #sfs = SequentialFeatureSelector(estimator, n_features_to_select=i, direction=direction, cv=5, n_jobs=5, scoring='f1_macro')
                #sfs = SequentialFeatureSelector(estimator, n_features_to_select=i, direction=direction, cv=4, n_jobs=4, scoring='f1_macro')
                #sfs = SequentialFeatureSelector(estimator, n_features_to_select=i, direction=direction, cv=kfoldobj, n_jobs=5, scoring='f1_macro')
                sfs = SequentialFeatureSelector(estimator, n_features_to_select=i, direction=direction, cv=kfoldinner, n_jobs=4, scoring='f1_macro')

                logger.debug(f'running SFS {direction} for {i} features')

                tmp['fold'] = kf
                tmp['fs_t1'] = time.time()
                #sfs.fit(xtr, ytr)
                sfs.fit(kfoldinner.xdata, kfoldinner.ydata)
                
                tmp['fs_t2'] = time.time()
                tmp['fs_time'] = tmp['fs_t2'] - tmp['fs_t1']

                # sffs = sfs.get_support()
                features = sfs.get_support(True)

                logger.debug(f'inner folds cross-validation (nested) - evaluate4')
                t8 = time.time()
                #innerfold = evaluate3(features, xtr, ytr)

                # use 4 inner folds, per well
                #innerfold = evaluate4(features, xtr, ytr, kfoldinner)
                innerfold = evaluate4(features, kfoldinner.xdata, kfoldinner.ydata, cv=kfoldinner)
                t9 = time.time()

                #breakpoint()
                innerf1 = np.mean(innerfold['test_f1_macro'])

                tmp['inner_valid_time'] = t9-t8
                iii = 0
                for i_i in range(5):
                    if i_i == kf:
                        tmp[f'inner_fold{i_i}_valid_f1macro'] = ''
                        continue
                    tmp[f'inner_fold{i_i}_valid_f1macro'] = innerfold['test_f1_macro'][iii]
                    iii = iii + 1

                #tmp['inner_fold0_valid_f1macro'] = innerfold['test_f1_macro'][0]
                #tmp['inner_fold1_valid_f1macro'] = innerfold['test_f1_macro'][1]
                #tmp['inner_fold2_valid_f1macro'] = innerfold['test_f1_macro'][2]
                #tmp['inner_fold3_valid_f1macro'] = innerfold['test_f1_macro'][3]
                #tmp['inner_fold4_valid_f1macro'] = innerfold['test_f1_macro'][4]
                tmp['inner_valid_f1macro'] = innerf1
                tmp['inner_valid_accuracy'] = np.mean(innerfold['test_accuracy'])
                tmp['inner_scores'] = str(innerfold)

                

                classif = classifier['model'](**classifier['default'])

                tmp['selected_features_str'] = str(features)
                tmp['selected_features'] = features

                tmp['train_t1'] = time.time()
                classif.fit(xtr[:, features], ytr)
                tmp['train_time'] = time.time() - tmp['train_t1']

                tmp['f1_macro'] = 0.0
                tmp['accuracy'] = 0.0
                f1mac = 0.0
                acc = 0.0


                #logger.debug(f'skipping training and testing for now')

                #"""

                try:
                    ypred = classif.predict(xte[:, features])

                    try:
                        f1mac = f1_score(yte, ypred, average="macro")
                        tmp['f1_macro'] = f1mac
                    except Exception as e3:
                        print(e3)

                    try:
                        acc = accuracy_score(yte, ypred)
                        tmp['accuracy'] = acc
                    except Exception as eacc:
                        print(eacc)

                except Exception as e:
                    print(e)
                #"""

                logger.debug(f'  mean f1 macro (inner) {innerf1:.4f}')
                logger.debug(f'  selected {str(features)}')
                logger.debug(f'  f1 macro (outer) {f1mac:.4f}')

                resultdata.append(tmp)

            frame = pd.DataFrame(data=resultdata)
            frame.to_excel(f'{params.experiment}_sfs_{direction}.xlsx')

        # hybrid
        if 0 > 1:
            for kf, data in enumerate(adata):
                logger.debug(f'outer fold {kf}')
                tmp = {}
                tmp['nof'] = 0
                xtr, ytr, xte, yte = data


                # =====================================================================
                # hybrid
                # =====================================================================

                logger.debug(f"ranking {params.ranking} ...")
                if params.ranking == "SVM":
                    features, rank = ranking_svm(xtr, ytr)
                elif params.ranking == "RF":
                    features, rank = ranking_rf(xtr, ytr)
                elif params.ranking == "ANOVA":
                    features, rank = ranking_anova(xtr, ytr)
                elif params.ranking == "CHI2":
                    features, rank = ranking_chi2()
                elif params.ranking == "LASSO":
                    features, rank = ranking_lasso(xtr, ytr)
                elif params.ranking == "MI":
                    features, rank = ranking_mi(xtr, ytr)

                logger.debug(f"scores {str(rank)}")
                logger.debug(f"feature rank sorted best to worst")
                logger.debug(f"  {str(features)}")

                # features = sorted(n)
                factor = 0.99
                maxfeatures = int(len(features) * 0.2)
                subset = []
                bestvalue = -1
                bestsubset = []
                currentvalue = -1
                currentsubset = []

                # breakpoint()

                iter_ = 0
                data = []
                while len(features) > 0:
                    iter_ = iter_ + 1
                    # feat = features.pop(ranklist.pop(0))
                    feat = features.pop(0)
                    subset.append(feat)
                    value = evaluate(subset, xtr, ytr, cfg=cfg, clf=clf)
                    if value > currentvalue * factor:
                        currentvalue = value
                        currentsubset = subset[:]
                    else:
                        logger.debug(f"{iter_:3d} feature {feat} nao melhorou, descartar")
                        #currentsubset = subset[:-1]
                        subset = subset[:-1]

                    if currentvalue > bestvalue:
                        bestvalue = currentvalue
                        bestsubset = currentsubset[:]
                    else:
                        if len(currentsubset) > maxfeatures:
                            currentvalue = bestvalue
                            currentsubset = bestsubset

                    logger.debug(
                        f"{iter_:3d} current subset [{len(currentsubset):2d}, {currentvalue:.4f}] {str(currentsubset)}"
                    )

                    data.append({
                        'iter': iter_,
                        'metric': currentvalue,
                        'size': len(currentsubset),
                        'try': feat,
                    })

                frame = pd.DataFrame(data=data)
                frame.to_excel(f"{params.experiment}_fold{kf}_hybrid.xlsx")


                features = currentsubset

                classif = classifier['model'](**classifier[cfg])


                tmp['selected_features_str'] = str(features)
                tmp['selected_features'] = features

                tmp['train_t1'] = time.time()
                classif.fit(xtr[:, features], ytr)
                tmp['train_time'] = time.time() - tmp['train_t1']

                try:
                    ypred = classif.predict(xte[:, features])
                    #yppred = np.sum(ypred)
                    #ynpred = ytest.shape[0] - yppred

                    try:
                        f1mac = f1_score(yte, ypred, average="macro")
                        tmp['f1_macro'] = f1mac
                    except Exception as e3:
                        print(e3)

                    try:
                        acc = accuracy_score(yte, ypred)
                        tmp['accuracy'] = acc
                    except Exception as eacc:
                        print(eacc)

                except Exception as e:
                    logger.exception(e)
                    print(e)

                logger.debug(f'  selected {str(features)}')
                logger.debug(f'  mean f1 macro {f1mac:.4f}')

                resultdata.append(tmp)

            frame = pd.DataFrame(data=resultdata)
            frame.to_excel(f'{params.experiment}_{params.ranking}.xlsx')

    except Exception as e000:
        logger.exception(e000)
        print(e000)


def singlefold(round_: int, fold: int, params: P, *args, **kwargs) -> None:
    """
    Run one fold.

    It can be executed in parallel.
    """
    logging.captureWarnings(True)
    logger = logging.getLogger(f"fold{fold}")
    formatter = logging.Formatter(params.logformat)
    fh = logging.FileHandler(f"{params.experiment}_feature.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    logger.debug(f"round {round_}")
    logger.debug(f"fold {fold}")

    # 0 => timestamp
    # 1 "P-PDG",
    # 2 "P-TPT",
    # 3 "T-TPT",
    # 4 "P-MON-CKP",
    # 5 "T-JUS-CKP",
    # 6 "P-JUS-CKGL",
    # 7 "T-JUS-CKGL",
    # 8 "QGL",
    # 9 => class label
    # 6, 7, 8 are gas lift related
    # usecols = [1, 2, 3, 4, 5]
    # usecols = params.usecols

    classifiers = get_classifiers(params.classifierstr)

    # try:
    #     xtrainneg, xtrainpos, xtestneg, xtestpos, xfn, xfp = params.read_and_split(
    #         fold, round_, params
    #     )
    # except Exception as e000:
    #     print(e000)
    #     logger.exception(e000)

    # x_outer_train, y_outer_train = vertical_split_bin(xtrainneg, xtrainpos)
    # x_outer_test, y_outer_test = vertical_split_bin(xtestneg, xtestpos)

    # mask = np.any(
    #     np.isnan(x_outer_train)
    #     | (x_outer_train > np.finfo(np.float32).max)
    #     | np.isinf(x_outer_train)
    #     | ~np.isfinite(x_outer_train),
    #     axis=1,
    # )

    # x_outer_train = x_outer_train[~mask]
    # y_outer_train = y_outer_train[~mask]

    # mask = np.any(
    #     np.isnan(x_outer_test)
    #     | (x_outer_test > np.finfo(np.float32).max)
    #     | np.isinf(x_outer_test)
    #     | ~np.isfinite(x_outer_test),
    #     axis=1,
    # )

    # x_outer_test = x_outer_test[~mask]
    # y_outer_test = y_outer_test[~mask]

    # x_data = np.concatenate([x_outer_train, x_outer_test], axis=0)
    # y_data = np.concatenate([y_outer_train, y_outer_test], axis=0)

    # scalerafter = StandardScaler()
    # scalerafter = MinMaxScaler(feature_range=(0.0, 1.0))
    # scalerafter.fit(x_outer_train)
    # x_outer_train = scalerafter.transform(x_outer_train)
    # x_outer_test = scalerafter.transform(x_outer_test)
    # scalerafter.fit(x_data)
    # x_data = scalerafter.transform(x_data)

    # logger.debug(f"finished filter, concat, scaling")

    # hybrid ranking wrapper feature selection
    # hybrid_ranking_wrapper(x_data, y_data, params, logger)

    # # genetic algorithm feature selection
    # paramdict = vars(params)
    # paramdict["obj"] = params
    # paramdict["verbose"] = 2
    # paramdict["dimension"] = x_data.shape[1]
    # paramdict["population"] = 100
    # paramdict["max_generations"] = 200
    # paramdict["mutation_rate"] = 0.01
    # paramdict["crossover_rate"] = 0.99
    # paramdict["crossover_alpha"] = 0.25
    # # paramdict['max_func_eval'] = 1000
    # paramdict["best_to_keep"] = 2

    # paramdict["classifiercfg"] = "default"
    # # paramdict['classifiercfg'] = 'best'

    # # breakpoint()
    # garesult = ga_optimizer(objective1, paramdict)
    # frame = pd.DataFrame(data=garesult["generations"])
    # frame.to_excel(f"ga_generations_{paramdict['experiment']}.xlsx")
    # logger.debug(garesult)
    # bestv = objective1(garesult["best_vector"], paramdict)
    # logger.debug(garesult["best_vector"])
    # logger.debug(bestv)
    # logger.debug(f"F1 Macro = {bestv['objective']:.4f}")

    # sys.exit(0)

    # psoresult = pso_optimizer(objective1, paramdict)
    # print(psoresult)
    # bestv = objective2(psoresult['best_vector'], paramdict)
    # print(bestv)
    # print(f"F1 Macro = {bestv['objective']:.4f}")
    # sys.exit(0)

    ckf = CustomKFold(params=params, random_state=1)
    clf = "RF"

    # for idx, f2 in zip(ckf.split(), ckf.split_()):
    #     xtrain1 = ckf.xdata[idx[0], :]
    #     xtrain2 = f2[0]
    #     xtest1 = ckf.xdata[idx[1], :]
    #     xtest2 = f2[2]

    #     print(xtest1.shape, xtest2.shape)
    #     if np.all(xtrain1 == xtrain2) and np.all(xtest1 == xtest2):
    #         print('ok')       

    # breakpoint()

    # if params.featureselection >= 0:
    try:
        classifier = get_classifiers([clf]).get(clf)
        # estimator = classifier['model'](**classifier['best'])
        estimator = classifier['model'](**classifier['default'])
        # logger.debug(str(estimator))
        # ['best']
        # cknn = get_classifiers(['1NN']).get('1NN')
        # knn = classifier['model'](**classifier['best'])
        # knn = classifier['model'](**classifier['default'])

        def evaluate(features, clf="RF", cfg="default"):
            classifier = classifiers[clf]
            estimator = classifier["model"](**classifier[cfg])
            scores = cross_validate(
                estimator,
                ckf.xdata[:, features],
                ckf.ydata,
                cv=ckf,
                n_jobs=5,
                scoring=["f1_macro", "accuracy"],
                error_score="raise",
            )
            return np.mean(scores["test_f1_macro"])

        # logger.debug('SelectKBest')
        # fsf = SelectKBest(f_classif, k=1)
        # fsf.fit(x_data, y_data)
        # fsfs = fsf.get_support()
        # fsf_scores = fsf.scores_
        # fsf_pvalues = fsf.pvalues_
        # logger.debug('FS f_classif' + str(fsf.get_support()))

        # logger.debug('SelectKBest chi2')
        # fsc = SelectKBest(chi2, k=1)
        # fsc.fit(x_data, y_data)
        # fscs = fsc.get_support()
        # print('FS chi2', fsc.get_support())

        # fsr = SelectKBest(estimator, k=1)
        # fsr.fit(x_data, y_data)
        # print('FS RF', fsr.get_support())

        # logger.debug(f"SequentialFeatureSelector - forward - {clf}")
        # sfs = SequentialFeatureSelector(estimator, n_features_to_select=None, direction='forward', cv=ckf, n_jobs=5, scoring='f1_macro')
        # sfs.fit(ckf.xdata, ckf.ydata)
        # sffs = sfs.get_support()
        # sffsi = sfs.get_support(True)
        # logger.debug(f'selected {str(sffs)} ')
        # logger.debug(f'selected {str(sffsi)}')
        # metric0 = 0.0
        direction = 'backward'
        logger.debug(f'sequential feature selection {direction}')
        data = []
        for i in range(54, 1, -1):
            tmp = {}
            tmp['nof'] = i
            logger.debug(f'try selecting {i} features')
            estimator = classifier['model'](**classifier['default'])

            # forward
            # sfs = SequentialFeatureSelector(estimator, n_features_to_select=i, direction='forward', cv=ckf, n_jobs=5, scoring='f1_macro')

            # backward
            sfs = SequentialFeatureSelector(estimator, n_features_to_select=i, direction=direction, cv=ckf, n_jobs=5, scoring='f1_macro')

            tmp['t1'] = time.time()
            sfs.fit(ckf.xdata, ckf.ydata)
            tmp['t2'] = time.time()

            sffs = sfs.get_support()
            sffsi = sfs.get_support(True)

            metric = evaluate(sffs)
            tmp['f1'] = metric

            # logger.debug(f'  selected {str(sffs)} ')
            logger.debug(f'  selected {str(sffsi)}')
            logger.debug(f'  mean f1 macro {metric:.4f}')

            # if metric <= metric0:
            #     print('optimum reached')
            #     break
            # metric0 = metric
            data.append(tmp)

        frame = pd.DataFrame(data=data)
        frame.to_excel(f'{params.experiment}_sfs_{direction}.xlsx')

        # breakpoint()

        # estimator = classifier["model"](**classifier["default"])
        # # sfs = SequentialFeatureSelector(estimator, n_features_to_select=1, direction='forward', cv=5, n_jobs=5, scoring='accuracy')
        # sfs = SequentialFeatureSelector(
        #     estimator,
        #     n_features_to_select=None,
        #     direction="forward",
        #     cv=5,
        #     n_jobs=5,
        #     scoring="f1_macro",
        # )
        # sfs.fit(x_data, y_data)
        # sffs = sfs.get_support()
        # sffsi = sfs.get_support(True)
        # logger.debug(f"support {str(sffs)}")
        # logger.debug(f"selected {str(sffsi)}")
        # for kf, f in enumerate(sffs):
        #    logger.debug(f'Feature {kf:2d} {f}')
        # logger.debug('SFS selected:' + str(sfs.get_support()))
        # logger.debug('SFS params:' + str(sfs.get_params()))

        # logger.debug('SequentialFeatureSelector - backward')
        # sfs = SequentialFeatureSelector(estimator, n_features_to_select=1, direction='backward', cv=5, n_jobs=1, scoring='f1_macro')
        # sfs.fit(x_data, y_data)
        # sfbs = sfs.get_support()

        # recursive
        # logger.debug("Recursive Feature Elimination")
        # rfe = RFE(estimator=estimator, n_features_to_select=None, step=1)
        # rfe.fit(x_data, y_data)
        # rfes = rfe.get_support()
        # rfesi = rfe.get_support(True)
        # logger.debug(f"support {str(rfes)}")
        # logger.debug(f"selected {str(rfesi)}")

        # pca = PCA(n_components=1, copy=True)
        # pca.fit(x_data, y_data)
        # logger.debug(f'{str(pca.components_)}')
        # logger.debug(f'{str(pca.explained_variance_)}')

        # ipca = IncrementalPCA(n_components=1, copy=True)
        # ipca.fit(x_data, y_data)
        # logger.debug(f'{str(ipca.components_)}')
        # logger.debug(f'{str(ipca.explained_variance_)}')

        # for c in range(len(params.usecols)):
        #     for k in range(params.nfeaturesvar):
        #         ck = c * params.nfeaturesvar + k
        #         logger.debug(
        #             f"sensor={c} {params.datasetcols[c+1]:10s} "
        #             f"feature={ck:2d} {params.featurefunc[k]:14s}   "
        #             f"SFF use={str(sffs[ck]):5s}  "
        #             # f'SFB use={str(sfbs[ck]):5s}  '
        #             # f'FSF use={str(fsfs[ck]):5s} {fsf.scores_[ck]:.4f}  '
        #             # f'FSC use={str(fscs[ck]):5s} {fsc.scores_[ck]:.4f}  '
        #             f"RFE use={str(rfes[ck]):5s}  "
        #         )

        # breakpoint()
    except Exception as e:
        # stackprinter.show()
        # logger.error(stackprinter.format())
        print_exc()

    sys.exit(1)


class CustomKFold(KFold):
    """
    CustomKFold returns INDEXES.
    """
    def __init__(
        self, n_splits=5, shuffle=False, random_state=None, params=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.params = params

        xdata = []
        ydata = []
        self.splits = []
        round_ = 1

        idxa = 0
        idxb = 0
        for fold in range(self.n_splits):
            try:
                xtrainneg, xtrainpos, xtestneg, xtestpos, _, _ = params.read_and_split(
                    fold, round_, self.params
                )
            except Exception as e000:
                print(e000)
                raise e000
            
            xtrain, y_train =  vertical_split_bin(xtrainneg, xtrainpos)
            xtest, y_test = vertical_split_bin(xtestneg, xtestpos)

            mask = np.any(
                np.isnan(xtrain)
                | (xtrain > np.finfo(np.float32).max)
                | np.isinf(xtrain)
                | ~np.isfinite(xtrain),
                axis=1,
            )

            xtrain = xtrain[~mask]
            y_train = y_train[~mask]

            mask = np.any(
                np.isnan(xtest)
                | (xtest > np.finfo(np.float32).max)
                | np.isinf(xtest)
                | ~np.isfinite(xtest),
                axis=1,
            )

            xtest = xtest[~mask]
            y_test = y_test[~mask]

            self.splits.append(
                (np.arange(idxa, idxa + xtrain.shape[0]), np.arange(idxa + xtrain.shape[0], idxa + xtrain.shape[0] + xtest.shape[0]))
            )

            idxa = idxa + xtrain.shape[0] + xtest.shape[0]

            scalerafter = StandardScaler()
            scalerafter.fit(xtrain)
            xtrain = scalerafter.transform(xtrain)
            xtest = scalerafter.transform(xtest)

            xdata.append(xtrain)
            xdata.append(xtest)

            ydata.append(y_train)
            ydata.append(y_test)

        self.xdata = np.concatenate(xdata, axis=0)
        self.ydata = np.concatenate(ydata, axis=0)

    def split(self, X=None, y=None, groups=None):
        """
        This method is compatible with Scikit-learn API.
        """
        for fold in range(len(self.splits)):
            yield self.splits[fold]

    def index_splits(self, X=None, y=None):
        return self.split(X, y)

    def data_splits(self, X=None, y=None):
        # round_ = self.random_state
        round_ = 1
        for fold in range(self.n_splits):
            try:
                (
                    xtrainneg,
                    xtrainpos,
                    xtestneg,
                    xtestpos,
                    xfn,
                    xfp,
                ) = self.params.read_and_split(fold, round_, self.params)
            except Exception as e000:
                print(e000)
                raise e000

            x_outter_train, y_outter_train = vertical_split_bin(xtrainneg, xtrainpos)
            x_outter_test, y_outter_test = vertical_split_bin(xtestneg, xtestpos)

            mask = np.any(
                np.isnan(x_outter_train)
                | (x_outter_train > np.finfo(np.float32).max)
                | np.isinf(x_outter_train)
                | ~np.isfinite(x_outter_train),
                axis=1,
            )

            x_outter_train = x_outter_train[~mask]
            y_outter_train = y_outter_train[~mask]

            mask = np.any(
                np.isnan(x_outter_test)
                | (x_outter_test > np.finfo(np.float32).max)
                | np.isinf(x_outter_test)
                | ~np.isfinite(x_outter_test),
                axis=1,
            )
            # mask = ~mask
            x_outter_test = x_outter_test[~mask]
            y_outter_test = y_outter_test[~mask]

            scalerafter = StandardScaler()
            scalerafter.fit(x_outter_train)
            x_outter_train = scalerafter.transform(x_outter_train)
            x_outter_test = scalerafter.transform(x_outter_test)

            yield x_outter_train, y_outter_train, x_outter_test, y_outter_test


class WellKFold(KFold):
    """
    WellKFold returns DATA.
    """
    def __init__(
        self, n_splits=5, shuffle=False, random_state=None, params=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.params = params
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X=None, y=None):
        # round_ = self.random_state
        round_ = 1
        for fold in range(self.n_splits):
            try:
                (
                    xtrainneg,
                    xtrainpos,
                    xtestneg,
                    xtestpos,
                    xfn,
                    xfp,
                ) = self.params.read_and_split(fold, round_, self.params)
            except Exception as e000:
                print(e000)
                raise e000

            x_outter_train, y_outter_train = vertical_split_bin(xtrainneg, xtrainpos)
            x_outter_test, y_outter_test = vertical_split_bin(xtestneg, xtestpos)

            mask = np.any(
                np.isnan(x_outter_train)
                | (x_outter_train > np.finfo(np.float32).max)
                | np.isinf(x_outter_train)
                | ~np.isfinite(x_outter_train),
                axis=1,
            )

            x_outter_train = x_outter_train[~mask]
            y_outter_train = y_outter_train[~mask]

            mask = np.any(
                np.isnan(x_outter_test)
                | (x_outter_test > np.finfo(np.float32).max)
                | np.isinf(x_outter_test)
                | ~np.isfinite(x_outter_test),
                axis=1,
            )
            # mask = ~mask
            x_outter_test = x_outter_test[~mask]
            y_outter_test = y_outter_test[~mask]

            scalerafter = StandardScaler()
            scalerafter.fit(x_outter_train)
            x_outter_train = scalerafter.transform(x_outter_train)
            x_outter_test = scalerafter.transform(x_outter_test)

            yield x_outter_train, y_outter_train, x_outter_test, y_outter_test


class InnerWellFold:
    """
    InnerWellFold return iterator for 4 INNER wells; returns INDEXES.
    """
    def __init__(
        self, outerfold, n_splits=5, shuffle=False, random_state=None, params=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.outerfold = outerfold
        self.params = params
        self.n_splits = n_splits

        xdata = []
        ydata = []
        self.splits = []
        round_ = 1

        idxa = 0
        idxb = 0
        for fold in range(self.n_splits):
            #if fold == outerfold:
            #    continue
            try:
                xtrainneg, xtrainpos, xtestneg, xtestpos, _, _ = params.read_and_split(
                    fold, round_, self.params
                )
            except Exception as e000:
                print(e000)
                breakpoint()
                raise e000
            
            xtrain, y_train =  vertical_split_bin(xtrainneg, xtrainpos)
            xtest, y_test = vertical_split_bin(xtestneg, xtestpos)

            mask = np.any(
                np.isnan(xtrain)
                | (xtrain > np.finfo(np.float32).max)
                | np.isinf(xtrain)
                | ~np.isfinite(xtrain),
                axis=1,
            )

            xtrain = xtrain[~mask]
            y_train = y_train[~mask]

            mask = np.any(
                np.isnan(xtest)
                | (xtest > np.finfo(np.float32).max)
                | np.isinf(xtest)
                | ~np.isfinite(xtest),
                axis=1,
            )

            xtest = xtest[~mask]
            y_test = y_test[~mask]

            self.splits.append(
                (np.arange(idxa, idxa + xtrain.shape[0]), np.arange(idxa + xtrain.shape[0], idxa + xtrain.shape[0] + xtest.shape[0]))
            )

            idxa = idxa + xtrain.shape[0] + xtest.shape[0]

            scalerafter = StandardScaler()
            scalerafter.fit(xtrain)
            xtrain = scalerafter.transform(xtrain)
            xtest = scalerafter.transform(xtest)

            xdata.append(xtrain)
            xdata.append(xtest)

            ydata.append(y_train)
            ydata.append(y_test)

        self.xdata = np.concatenate(xdata, axis=0)
        self.ydata = np.concatenate(ydata, axis=0)

    def split(self, X=None, y=None, groups=None):
        for fold in range(len(self.splits)):
            if fold == self.outerfold:
                continue
            #print(f'nested CV per well, inner fold {fold}')
            yield self.splits[fold]

    def index_splits(self, X=None, y=None):
        return self.split(X, y)

    def data_splits(self, X=None, y=None):
        # round_ = self.random_state
        raise Exception

    def get_n_splits(self, *args, **kwargs):
        return 4


class InnerWellFold2:
    def __init__(
        self, outerfold, n_splits=5, shuffle=False, random_state=None, params=None, *args, **kwargs
    ):
        self.outerfold = outerfold
        self.xdata = None
        self.ydata = None
        self.splits = []

    def split(self, X=None, y=None, groups=None):
        for fold in range(len(self.splits)):
            if fold == self.outerfold:
                continue
            #print(f'nested CV per well, inner fold {fold}')
            yield self.splits[fold]

    def get_n_splits(self, *args, **kwargs):
        return 4 


def foldfn(round_: int, fold: int, params: P) -> List[Dict]:
    """
    Run one fold.

    It can be executed in parallel.
    """
    logging.captureWarnings(True)
    logger = logging.getLogger(f"fold{fold}")
    formatter = logging.Formatter(params.logformat)
    fh = logging.FileHandler(f"{params.experiment}_fold{fold}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    logger.debug(f"round {round_}")
    logger.debug(f"fold {fold}")

    # 0 => timestamp
    # 1 "P-PDG",
    # 2 "P-TPT",
    # 3 "T-TPT",
    # 4 "P-MON-CKP",
    # 5 "T-JUS-CKP",
    # 6 "P-JUS-CKGL",
    # 7 "T-JUS-CKGL",
    # 8 "QGL",
    # 9 => class label
    # 6, 7, 8 are gas lift related
    # usecols = [1, 2, 3, 4, 5]
    # usecols = params.usecols

    classifiers = get_classifiers(params.classifierstr)

    # breakpoint()

    try:

        # ==============================================================================
        # Read data from disk and split in folds
        # ==============================================================================
        logger.debug(f"read and split data in folds")

        try:
            xtrainneg, xtrainpos, xtestneg, xtestpos, xfn, xfp = params.read_and_split(
                fold, round_, params
            )
        except Exception as e000:
            print(e000)
            raise e000

        x_outter_train, y_outter_train = vertical_split_bin(xtrainneg, xtrainpos)
        x_outter_test, y_outter_test = vertical_split_bin(xtestneg, xtestpos)

        if len(params.usecols) > 0:
            usecols = []
            # for c in params.usecols:
            #    for ck in range((c - 1) * params.nfeaturesvar, c * params.nfeaturesvar):
            #        usecols.append(ck)
            for c in range(len(params.usecols)):
                for ck in range(params.nfeaturesvar):
                    # cidx = params.nfeaturesvar * c + ck
                    cidx = len(params.featurefunc) * c + ck
                    usecols.append(cidx)
            print(
                "use measured variables",
                str(params.usecols),
                " keep features ",
                str(usecols),
            )
            logger.info(
                "use measured variables"
                + str(params.usecols)
                + " keep features "
                + str(usecols)
            )
            x_outter_train = x_outter_train[:, usecols]
            x_outter_test = x_outter_test[:, usecols]

        if isinstance(params.usefeatures, list) and len(params.usefeatures) > 0:
            logger.debug(f"use only features {str(params.usefeatures)}")
            x_outter_train = x_outter_train[:, params.usefeatures]
            x_outter_test = x_outter_test[:, params.usefeatures]
        else:
            # breakpoint()
            pass

        logger.debug(f"train neg={str(xtrainneg.shape)} pos={str(xtrainpos.shape)}")
        logger.debug(f"test  neg={str(xtestneg.shape)}  pos={str(xtestpos.shape)}")

        if 0 in xtestpos.shape or 0 in xtestneg.shape:
            breakpoint()
            raise Exception("dimension zero")

        if xtestpos.shape[0] > xtestneg.shape[0]:
            #
            # print('Binary problem with unbalanced classes: NEG is not > POS')
            logger.warn("Binary problem with unbalanced classes: NEG is not > POS")
            # raise Exception()

        logger.debug(
            f"shapes train={str(x_outter_train.shape)} test={str(x_outter_test.shape)}"
        )
        # ==============================================================================
        # After feature extraction, some NaN appear again in the arrays
        # ==============================================================================
        logger.debug(f"check NaN #1")
        mask = np.any(
            np.isnan(x_outter_train)
            | (x_outter_train > np.finfo(np.float32).max)
            | np.isinf(x_outter_train)
            | ~np.isfinite(x_outter_train),
            axis=1,
        )
        # breakpoint()
        logger.debug(f"NaN Mask size {np.count_nonzero(mask, axis=0)}")

        x_outter_train = x_outter_train[~mask]
        y_outter_train = y_outter_train[~mask]

        mask = np.any(
            np.isnan(x_outter_test)
            | (x_outter_test > np.finfo(np.float32).max)
            | np.isinf(x_outter_test)
            | ~np.isfinite(x_outter_test),
            axis=1,
        )
        # mask = ~mask
        x_outter_test = x_outter_test[~mask]
        y_outter_test = y_outter_test[~mask]

        logger.debug(f"check NaN #2")
        check_nan(x_outter_train, logger)

        logger.debug(
            f"shapes train={str(x_outter_train.shape)} test={str(x_outter_test.shape)}"
        )

        # ==============================================================================
        # Normalization
        # ==============================================================================
        logger.debug(f"normalization AFTER feature extraction")
        scalerafter = StandardScaler()
        scalerafter.fit(x_outter_train)
        x_outter_train = scalerafter.transform(x_outter_train)
        x_outter_test = scalerafter.transform(x_outter_test)
        # ==============================================================================

        # pca = PCA(n_components=1, copy=True)
        # pca.fit(x_outter_train, y_outter_train)
        # x_outter_train = pca.transform(x_outter_train)
        # x_outter_test = pca.transform(x_outter_test)

        resultlist = []

        for clf in classifiers:
            logger.debug(f"Classifier {clf}")
            if isinstance(classifiers[clf]["model"], str):
                model = eval(classifiers[clf]["model"])
            elif callable(classifiers[clf]["model"]):
                model = classifiers[clf]["model"]

            if params.gridsearch == 1:
                # raise Exception("not implemented")

                # inner = []

                logger.debug("Grid Search with Cross Validation (inner loop)")

                grid = classifiers[clf]["config"][0]
                if "random_state" in classifiers[clf]["config"][1]:
                    grid["random_state"] = [round_]

                kfcv = KFold(n_splits=4, shuffle=True, random_state=round_)
                gscv = GridSearchCV(
                    estimator=model(),
                    param_grid=grid,
                    cv=kfcv,
                    scoring=["f1_macro", "accuracy"],
                    n_jobs=1,
                    refit="f1_macro",
                )

                gscv.fit(x_outter_train, y_outter_train)
                # print(gscv.cv_results_.keys())
                # breakpoint()

                best_config = gscv.best_params_
                idx = gscv.best_index_
                logger.debug(gscv.cv_results_["rank_test_f1_macro"])
                logger.debug(f"best config: {idx:2d} " + str(best_config))
                # breakpoint()

                # ===============
                # FIM GRID SEARCH
                # ===============

            elif params.gridsearch == 2:

                logger.debug("Bayesian optimization over hyper parameters with Cross Validation (inner loop)")

                grid = classifiers[clf]["config"][0]
                if "random_state" in classifiers[clf]["config"][1]:
                    grid["random_state"] = [round_]

                kfcv = KFold(n_splits=4, shuffle=True, random_state=round_)
                wcv = InnerWellFold(outerfold=fold, params=params)

                gscv = BayesSearchCV(
                    estimator=model(),
                    search_spaces=classifiers[clf]["opt"],
                    n_iter=50,
                    scoring="f1_macro",
                    n_jobs=1,
                    cv=wcv,
                    #cv=kfcv,
                    random_state=round_,
                    #refit="f1_macro",
                    refit=True,
                    verbose=1,
                )

                #gscv.fit(x_outter_train, y_outter_train)
                _ = gscv.fit(wcv.xdata, wcv.ydata)
                # print(gscv.cv_results_.keys())
                # breakpoint()

                best_config = gscv.best_params_
                idx = gscv.best_index_
                logger.debug(gscv.cv_results_["rank_test_f1_macro"])
                logger.debug(f"best config: {idx:2d} " + str(best_config))
                logger.debug(f"best score {str(gscv.best_score_)}")
                # breakpoint()

                # ===============
                # FIM GRID SEARCH
                # ===============

            else:
                idx = 0
                best_config = classifiers[clf]["default"]

            if "random_state" in best_config:
                best_config["random_state"] = round_

            classif = None
            classif = model(**best_config)

            r1, r2, r3, r4 = train_test_binary(
                classif, x_outter_train, y_outter_train, x_outter_test, y_outter_test
            )
            y_outter_pred, ynpred, yppred, traintime = r1
            f1bin4, f1bin0, f1mic, f1mac, f1weigh, f1sam, p, r, acc = r2
            tn, fp, fn, tp = r3

            for exp in r4:
                logger.exception(exp)

            logger.info(
                f"Classifier {clf} acc={acc:.4f} f1mac={f1mac:.4f} f1bin4={f1bin4:.4f}  f1bin0={f1bin0:.4f}"
            )

            resultlist.append(
                vars(
                    Results(
                        class_="4",
                        experiment=params.experiment,
                        nfeaturestotal=0,
                        timestamp=np.int(f"{params.sessionts:%Y%m%d%H%M%S}"),
                        seed=round_,
                        foldoutter=fold,
                        foldinner=-1,
                        classifier=clf,
                        classifiercfg=idx,
                        classifiercfgs=len(classifiers[clf]["config"][0]),
                        f1binp=f1bin4,
                        f1binn=f1bin0,
                        f1micro=f1mic,
                        f1macro=f1mac,
                        f1weighted=f1weigh,
                        f1samples=f1sam,
                        precision=p,
                        recall=r,
                        accuracy=acc,
                        accuracy2=0.0,
                        timeinnertrain=0,
                        timeouttertrain=traintime,
                        positiveclasses="4",
                        negativeclasses="0",
                        features="",
                        nfeaturesvar=params.nfeaturesvar,
                        # postrainsamples=trainpositive.shape[0],
                        # negtrainsamples=trainnegative.shape[0],
                        # postestsamples=testpositive.shape[0],
                        # negtestsamples=testnegative.shape[0],
                        postrainsamples=0,
                        negtrainsamples=0,
                        postestsamples=0,
                        negtestsamples=0,
                        ynegfinaltrain=xtrainneg.shape[0],
                        yposfinaltrain=xtrainpos.shape[0],
                        ynegfinaltest=xtestneg.shape[0],
                        yposfinaltest=xtestpos.shape[0],
                        yposfinalpred=yppred,
                        ynegfinalpred=ynpred,
                        yfinaltrain=y_outter_train.shape[0],
                        yfinaltest=y_outter_test.shape[0],
                        yfinalpred=y_outter_pred.shape[0],
                        tp=tp,
                        tn=tn,
                        fp=fp,
                        fn=fn,
                        bestfeatureidx="",
                        bestvariableidx="",
                        featurerank="",
                        rankfeature="",
                    )
                )
            )

        return resultlist

    except Exception as efold:
        logger.exception(efold)
        print("="*78, file=sys.stdout)
        traceback.print_exc(file=sys.stdout)
        print("="*78, file=sys.stdout)
        print_exc(efold)
        print("="*78, file=sys.stdout)
        breakpoint()
        raise efold

    return []


def runexperiment(params, *args, **kwargs) -> None:
    """
    Run experiment - train, validation (optional) and test.
    """
    all_results_list = []
    partial = []

    # logger with default config
    # logging.config.dictConfig(get_logging_config())
    # logger = logging.getLogger()
    # logger.setLevel(logging.DEBUG)

    logger = logging.getLogger(__name__)
    formatter = logging.Formatter(params.logformat)
    fh = logging.FileHandler(f"{params.experiment}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    logger.info(params.name)
    logger.info("=" * 79)

    t0 = time.time()
    # logger.info(f'Started}')

    for round_ in range(1, params.nrounds + 1):

        # pipepine(round_, -1, params)
        # singlefold(round_, 1, params)
        # breakpoint()

        logger.info(f"Round {round_}")

        if params.njobs > 1:
            logger.debug(f"Running {params.njobs} parallel jobs")
            with Pool(processes=params.njobs) as p:
                # 'results' is a List of List
                results = p.starmap(
                    foldfn,
                    zip(
                        [round_] * params.nfolds,
                        range(params.nfolds),
                        [params] * params.nfolds,
                    ),
                )
            results_list_round = []
            for r in results:
                results_list_round.extend(r)
        else:
            logger.debug(f"Running single core")
            results_list_round = []
            for foldout in range(0, params.nfolds):
                results_list_round.extend(foldfn(round_, foldout, params))

        partial = pd.DataFrame(data=results_list_round)
        try:
            partial.to_excel(f"{params.experiment}_parcial_{round_}.xlsx", index=False)
        except Exception as e1:
            logger.exception(e1)

        all_results_list.extend(results_list_round)

    results = pd.DataFrame(data=all_results_list)
    results["e"] = params.experiment[:-1]
    results["case"] = params.experiment[-1]

    try:
        results.to_excel(f"{params.experiment}_final.xlsx", index=False)
    except Exception as e2:
        logger.exception(e2)

    try:
        # folds de treinamento
        markdown = str(
            pd.pivot_table(
                data=results[results["foldoutter"] >= 0],
                values=["f1macro", "ynegfinaltest", "yposfinaltest"],
                index=["classifier", "e"],
                aggfunc={
                    "f1macro": ["mean", "std"],
                    "ynegfinaltest": "max",
                    "yposfinaltest": "max",
                    # "class_": "count"
                },
                columns=["case"],
            ).to_markdown(buf=None, index=True)
        )
        logger.debug("\n" + markdown)
    except:
        pass

    try:

        with open(f"{params.experiment}_final.md", "w") as f:
            f.writelines("\n")
            pd.pivot_table(
                data=results[results["foldoutter"] >= 0],
                values=["f1macro", "ynegfinaltest", "yposfinaltest", "class_"],
                index=["classifier", "e"],
                aggfunc={
                    "f1macro": ["mean", "std"],
                    "ynegfinaltest": "max",
                    "yposfinaltest": "max",
                    "class_": "count",
                },
                columns=["case"],
            ).to_markdown(buf=f, index=True)
            f.writelines("\n")
            f.writelines(
                # folds de "teste"
                pd.pivot_table(
                    data=results[results["foldoutter"] < 0],
                    values=["f1macro", "ynegfinaltest", "yposfinaltest"],
                    index=["classifier", "e"],
                    aggfunc={
                        "f1macro": ["mean", "std"],
                        "ynegfinaltest": "max",
                        "yposfinaltest": "max",
                    },
                    columns=["case"],
                ).to_markdown(buf=None, index=True)
            )
            f.writelines("\n\n")
            f.writelines("Round 1\n")
            pd.pivot_table(
                data=results[
                    (results["foldoutter"] >= 0)
                    & (results["foldinner"] < 0)
                    & (results["seed"] == 1)
                ],
                values=["f1macro", "ynegfinaltest", "yposfinaltest", "class_"],
                index=["classifier", "e"],
                aggfunc={
                    "f1macro": ["mean", "std"],
                    "ynegfinaltest": "max",
                    "yposfinaltest": "max",
                    "class_": "count",
                },
                columns=["case"],
            ).to_markdown(buf=f, index=True)
            f.writelines("\n\n")

            classifiers = get_classifiers(params.classifierstr)
            for ck, cfg in enumerate(
                get_config_combination_list(classifiers["RF"]["config"][0])
            ):
                f.write(f"{ck:2d} = {str(cfg)} \n")
            f.write(f"default = {str(classifiers['RF']['default'])} \n")

    except Exception as e3:
        logger.exception(e3)

    logger.debug(f"finished in {humantime(seconds=(time.time()-t0))}")


#@dataclass(frozen=False)
class DefaultParams(Generic[P]):
    """
    Experiment configuration (and model hyperparameters)
    DataClass offers a little advantage over simple dictionary in that it checks if
    the parameter actually exists. A dict would accept anything.
    """

    name: str = ""
    experiment: str = ""
    nrounds: int = 1
    nfolds: int = 5
    njobs: int = 1
    windowsize: int = 900
    stepsize: int = 900
    gridsearch: int = 0
    classifierstr: str = "1NN,3NN,QDA,LDA,GNB,RF,ZERORULE"
    usecolsstr: str = "1,2,3,4,5"
    #usecols: list = field(default_factory=list)
    featurefunc: str = ""
    nfeaturesvar: int = 11
    hostname: str = socket.gethostname()
    ncpu: int = psutil.cpu_count()
    #datasetcols: list = field(default_factory=list)
    tzsp = tz.gettz("America/Sao_Paulo")
    # logformat: str = "%(asctime)s %(levelname)-8s  %(name)-12s %(funcName)-30s %(lineno)-5d %(message)s"
    logformat: str = "%(asctime)s %(levelname)-8s %(module)-12s %(funcName)-30s %(lineno)-5d %(message)s"
    shuffle: bool = True
    skipwellstr: str = ""
    usefeatures: str = ""
    ranking: str = "RF"
    # usefeaturefunc: int = 0
    fsdirection: str = ''
    featureselection: bool = False
    read_and_split = None

    def __init__(self, *args, **kwargs):
        self.usecols = []
        self.datasetcols = []
        for k, i in kwargs.items():
            setattr(self, k, i)


    def __post_init__(self):

        self.njobs = max(min(self.nfolds, self.njobs, self.ncpu), 1)
        self.skipwell = self.skipwellstr.split(",")
        self.sessionts = datetime.now(tz=self.tzsp)
        if isinstance(self.classifierstr, str):
            self.classifiers = self.classifierstr.split(",")
        elif isinstance(self.classifierstr, tuple):
            self.classifiers = list(self.classifierstr)
        self.usecols = self._none_int_str_tuple_list(self.usecolsstr)

        self.datasetcols = [
            "timestamp",
            "P-PDG",
            "P-TPT",
            "T-TPT",
            "P-MON-CKP",
            "T-JUS-CKP",
            "P-JUS-CKGL",
            "T-JUS-CKGL",
            "QGL",
            "class",
        ]

        self.featurefunc = [
            "max",
            "mean",
            "median",
            "min",
            "std",
            "var",
            "kurtosis",
            "skewness",
            "sampleentropy",
            "25%",
            "75%",
        ]
        self.usefeatures = self._none_int_str_tuple_list(self.usefeatures)

    def _none_int_str_tuple_list(self, var: Union[None, str, int, Tuple, List]) -> List:

        if var is None:
            return []

        if isinstance(var, list):
            return var

        elif isinstance(var, str):
            var = var.replace(" ", "")
            if len(var) > 0:
                if "-" in var:
                    s = var.split("-")
                    return list(range(int(s[0]), int(s[1])))
                else:
                    return list(map(int, var.split(",")))
            else:
                return []

        elif isinstance(var, int):
            return [var]

        elif isinstance(var, tuple):
            return list(var)


def cv_fold(clf, cfg, xtrain, ytrain, xtest, ytest, params) -> float:
    classifiers = get_classifiers(params.classifierstr)
    tmpscores = np.zeros(5, dtype=float)
    classifier = classifiers[clf]

    estimator = classifier["model"](**classifier[cfg])
    estimator.fit(xtrain, ytrain)
    ypred = estimator.predict(xtest)

    try:
        f1macro = f1_score(ytest, ypred, average="macro")
    except Exception as e3:
        print(e3)
        raise e3
    # tmpscores[dk] = f1macro

    return f1macro


def hybrid_ranking_wrapper(params, *args, **kwargs):
    round_ = 1
    fold = 0
    logging.captureWarnings(True)
    logger = logging.getLogger(f"fold{fold}")
    formatter = logging.Formatter(params.logformat)
    fh = logging.FileHandler(f"{params.experiment}_HRW_feature.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    logger.debug(f"round {round_}")
    logger.debug(f"fold {fold}")

    # 0 => timestamp
    # 1 "P-PDG",
    # 2 "P-TPT",
    # 3 "T-TPT",
    # 4 "P-MON-CKP",
    # 5 "T-JUS-CKP",
    # 6 "P-JUS-CKGL",
    # 7 "T-JUS-CKGL",
    # 8 "QGL",
    # 9 => class label
    # 6, 7, 8 are gas lift related
    # usecols = [1, 2, 3, 4, 5]
    # usecols = params.usecols

    n = len(params.usecols) * params.nfeaturesvar

    classifiers = get_classifiers(params.classifierstr)
    # classifier = classifiers['RF']
    # estimator = classifier['model'](**classifier['default'])
    # breakpoint()

    ckf = CustomKFold(params=params, random_state=1)
    clf = "RF"
    cfg = "default"
    classifier = classifiers[clf]

    def ranking_svm():
        """
        univariate feature estimate
        """
        rank = np.zeros(n, dtype=float)
        for it in range(n):
            # estimator = classifier['model'](**classifier['default'])
            # scores = cross_validate(estimator, x_data[:, [it]], y_data, cv=5, n_jobs=5, scoring=['f1_macro', 'accuracy'], error_score='raise')
            # print(scores)
            tmpscores = np.zeros(5, dtype=float)
            for dk, data in enumerate(ckf.data_splits()):
                xtrain, ytrain, xtest, ytest = data
                classifier = classifiers["SVM"]
                estimator = classifier["model"](**classifier["default"])
                estimator.fit(xtrain[:, [it]], ytrain)
                ypred = estimator.predict(xtest[:, [it]])
                try:
                    f1mac = f1_score(ytest, ypred, average="macro")
                except Exception as e3:
                    print(e3)
                tmpscores[dk] = f1mac

            # rank[it] = np.mean(scores['test_f1_macro'])
            rank[it] = np.mean(tmpscores)
        return list(np.argsort(rank)[::-1]), rank

    def ranking_rf():
        rank = np.zeros(n, dtype=float)
        for it in range(n):
            #estimator = classifier['model'](**classifier['default'])
            # scores = cross_validate(estimator, x_data[:, [it]], y_data, cv=5, n_jobs=5, scoring=['f1_macro', 'accuracy'], error_score='raise')
            # print(scores)
            #rank[it] = evaluate([it], cfg="default", clf="RF")
            #evaluate_hrw_4(subset, x, y, estimator, cv=None)
            # tmpscores = np.zeros(5, dtype=float)
            # for dk, data in enumerate(ckf.data_splits()):
            #     xtrain, ytrain, xtest, ytest = data
            #     estimator = classifier['model'](**classifier['default'])
            #     estimator.fit(xtrain[:, [it]], ytrain)
            #     ypred = estimator.predict(xtest[:, [it]])
            #     try:
            #         f1mac = f1_score(ytest, ypred, average="macro")
            #     except Exception as e3:
            #         excp.append(e3)
            #     tmpscores[dk] = f1mac

            # #rank[it] = np.mean(scores['test_f1_macro'])
            # rank[it] = np.mean(tmpscores)
            classifier = classifiers["RF"]
            estimator = classifier["model"](**classifier[cfg])
            scores = cross_validate(
                estimator,
                ckf.xdata[:, it],
                ckf.ydata,
                cv=ckf,
                #n_jobs=5,
                n_jobs=1,
                scoring=["f1_macro", "accuracy"],
                error_score="raise",
            )
            #s = np.mean(scores["test_f1_macro"])
            rank[it] = np.mean(scores["test_f1_macro"])

        return list(np.argsort(rank)[::-1]), rank

    def ranking_anova():
        fsf = SelectKBest(f_classif, k="all")
        fsf.fit(ckf.xdata, ckf.ydata)
        return list(np.argsort(fsf.scores_)[::-1]), fsf.scores_

    def ranking_mi():
        # mi = SelectKBest(chi2, k='all')
        # mi.fit(x_data, y_data)
        miarray = mutual_info_classif(
            ckf.xdata, ckf.ydata, discrete_features=False, n_neighbors=3, copy=True, random_state=1
        )
        # return list(np.argsort(mi.scores_)[::-1]), fsc.scores_
        return list(np.argsort(miarray)[::-1]), miarray

    def ranking_lasso():
        rank = np.zeros(n, dtype=float)
        coef = np.zeros((5, 55), dtype=float)
        for f in range(5):
            for dk, data in enumerate(ckf.data_splits()):
                xtrain, ytrain, xtest, ytest = data
                estimator = Lasso()
                estimator.fit(xtrain, ytrain)
                # ypred = estimator.predict(xtest[:, [it]])
                # try:
                #    f1mac = f1_score(ytest, ypred, average="macro")
                # except Exception as e3:
                #    excp.append(e3)
                # tmpscores[dk] = f1mac
                coef[f, :] = np.abs(estimator.coef_)

        rank = np.mean(coef, axis=0)
        return list(np.argsort(rank)[::-1]), rank

    def evaluate(s, cfg="default", clf="RF"):
        tmpscores = np.zeros(5, dtype=float)
        classifier = classifiers[clf]
        a = []
        b = []
        c = []
        d = []
        e = []
        f = []
        for dk, data in enumerate(ckf.data_splits()):
            xtrain, ytrain, xtest, ytest = data
            a.append(xtrain[:, s])
            b.append(ytrain)
            c.append(xtest[:, s])
            d.append(ytest)
            e.append(clf)
            f.append(cfg)

        with Pool(processes=params.njobs) as p:
            # 'results' is a List of Dict
            results = p.starmap(cv_fold, zip(e, f, a, b, c, d, [params] * 5))
            tmpscores = np.array(results)

        # print('custom')
        # tmpscores2 = np.zeros(5, dtype=float)
        # for dk, (tr, te) in enumerate(ckf.split()):

        #     xtrain = ckf.xdata[tr, :]
        #     ytrain = ckf.ydata[tr]

        #     xtest = ckf.xdata[te, :]
        #     ytest = ckf.ydata[te]

        #     print(xtrain.shape, xtest.shape)

        #     #xtrain, ytrain, xtest, ytest = data
        #     estimator = classifier['model'](**classifier[cfg])
        #     estimator.fit(xtrain[:, s], ytrain)
        #     ypred = estimator.predict(xtest[:, s])
        #     try:
        #         f1mac = f1_score(ytest, ypred, average="macro")
        #     except Exception as e3:
        #         excp.append(e3)
        #     tmpscores2[dk] = f1mac

        # print(np.mean(tmpscores))
        # print(np.mean(tmpscores2))

        return np.mean(tmpscores)

    def evaluate2(s, cfg="default"):
        classifier = classifiers["RF"]
        estimator = classifier["model"](**classifier[cfg])
        scores = cross_validate(
            estimator,
            ckf.xdata[:, s],
            ckf.ydata,
            cv=ckf,
            n_jobs=5,
            scoring=["f1_macro", "accuracy"],
            error_score="raise",
        )
        return np.mean(scores["test_f1_macro"])

    def evaluate_hrw_4(subset, x, y, estimator, cv=None):
        logger.debug(f'inner cross-validation ')
        scores = cross_validate(
            estimator, 
            x[:, subset], 
            y, 
            cv=cv, 
            n_jobs=1, 
            scoring=["f1_macro", "accuracy"],
            error_score="raise",
        )
        return scores
    # breakpoint()

    logger.debug(f"ranking {params.ranking} ...")
    if params.ranking == "SVM":
        features, rank = ranking_svm()
    elif params.ranking == "RF":
        features, rank = ranking_rf()
    elif params.ranking == "ANOVA":
        features, rank = ranking_anova()
    elif params.ranking == "LASSO":
        features, rank = ranking_lasso()
    elif params.ranking == "MI":
        features, rank = ranking_mi()

    logger.debug(f"scores {str(rank)}")
    logger.debug(f"feature rank sorted best to worst")
    logger.debug(f"  {str(features)}")

    # features = sorted(n)
    factor = 0.99
    maxfeatures = int(len(features) * 0.2)
    subset = []
    bestvalue = -1
    bestsubset = []
    currentvalue = -1
    currentsubset = []

    # breakpoint()

    iter_ = 0
    data = []
    while len(features) > 0:
        iter_ = iter_ + 1
        # feat = features.pop(ranklist.pop(0))
        feat = features.pop(0)
        subset.append(feat)
        value = evaluate(subset, cfg=cfg, clf=clf)
        if value > currentvalue * factor:
            currentvalue = value
            currentsubset = subset[:]
        else:
            logger.debug(f"{iter_:3d} feature {feat} nao melhorou, descartar")
            #currentsubset = subset[:-1]
            subset = subset[:-1]

        if currentvalue > bestvalue:
            bestvalue = currentvalue
            bestsubset = currentsubset[:]
        else:
            if len(currentsubset) > maxfeatures:
                currentvalue = bestvalue
                currentsubset = bestsubset

        logger.debug(
            f"{iter_:3d} current subset [{len(currentsubset):2d}, {currentvalue:.4f}] {str(currentsubset)}"
        )

        data.append({
            'iter': iter_,
            'metric': currentvalue,
            'size': len(currentsubset),
            'try': feat,
        })

    frame = pd.DataFrame(data=data)
    frame.to_excel(f"{params.experiment}_hybrid.xlsx")

    logger.debug(f"current subset [{len(currentsubset)}] {str(currentsubset)}")
    logger.debug(f"current value  {str(currentvalue)}")
    logger.debug(f'retrain default     {evaluate(currentsubset, "default", clf):.4f}')
    logger.debug(f'retrain best config {evaluate(currentsubset, "best", clf):.4f}')

    if len(currentsubset) > maxfeatures:
        ckf = CustomKFold(params=params, random_state=1)
        logger.debug(f"running SFS backward ({clf} {cfg})")
        estimator = classifier["model"](**classifier[cfg])
        sfs = SequentialFeatureSelector(
            estimator,
            n_features_to_select=maxfeatures,
            direction="backward",
            #cv=5,
            cv=ckf,
            n_jobs=5,
            scoring="f1_macro",
        )
        sfs.fit(ckf.xdata[:, bestsubset], ckf.ydata)
        sfbs = sfs.get_support()

        newlist = []
        for c in range(len(bestsubset)):
            logger.debug(f"feature={c:2d}   " f"SFB use={str(sfbs[c]):5s}  ")
            if sfbs[c]:
                newlist.append(bestsubset[c])

        logger.debug("final refit after SBS")
        logger.debug(str(newlist))
        # scores = cross_validate(estimator, x_data[:, newlist], y_data, cv=5, n_jobs=5, scoring=['f1_macro', 'accuracy'], error_score='raise')
        # logger.debug(str(scores))
        logger.debug(f'retrain default     {evaluate(newlist, "default"):.4f}')
        logger.debug(f'retrain best config {evaluate(newlist, "best"):.4f}')

    sys.exit(0)


def hybrid_ranking_wrapper_nested(params, *args, **kwargs):
    round_ = 1
    
    logging.captureWarnings(True)
    logger = logging.getLogger(f"HRW_NESTED")
    
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(funcName)-30s %(lineno)-5d %(message)s")
    fh = logging.FileHandler(f"{params.experiment}_hrw_feature.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    n = len(params.usecols) * params.nfeaturesvar
    classifiers = get_classifiers(params.classifierstr)
    ckf = CustomKFold(params=params, random_state=1)
    clf = "RF"
    cfg = "default"
    classifier = classifiers[clf]

    wkf = WellKFold()

    logger.debug("")
    logger.info('=======================')
    logger.info(f"{params.experiment} nested")
    logger.info('=======================')

    def ranking_rf(n, x, y, ckf):
        rank = np.zeros(n, dtype=float)
        for it in range(0, n, 1):
            classifier = classifiers["RF"]
            estimator = classifier["model"](**classifier[cfg])
            scores = cross_validate(
                estimator,
                ckf.xdata[:, [it]],
                ckf.ydata,
                cv=ckf,
                #n_jobs=5,
                n_jobs=1,
                scoring=["f1_macro", "accuracy"],
                error_score="raise",
            )
            #s = np.mean(scores["test_f1_macro"])
            rank[it] = np.mean(scores["test_f1_macro"])
            logger.debug(f"ranking {it:02d} = {rank[it]:.4f}")

        return list(np.argsort(rank)[::-1]), rank

    def ranking_anova(x, y):
        fsf = SelectKBest(f_classif, k="all")
        #fsf.fit(ckf.xdata, ckf.ydata)
        fsf.fit(x, y)
        return list(np.argsort(fsf.scores_)[::-1]), fsf.scores_

    def ranking_mi(x, y):
        # mi = SelectKBest(chi2, k='all')
        # mi.fit(x_data, y_data)
        miarray = mutual_info_classif(
            #ckf.xdata, ckf.ydata, discrete_features=False, n_neighbors=3, copy=True, random_state=1
            x, y, discrete_features=False, n_neighbors=3, copy=True, random_state=1
        )
        # return list(np.argsort(mi.scores_)[::-1]), fsc.scores_
        return list(np.argsort(miarray)[::-1]), miarray

    def ranking_lasso():
        rank = np.zeros(n, dtype=float)
        coef = np.zeros((5, 55), dtype=float)
        for f in range(5):
            for dk, data in enumerate(ckf.data_splits()):
                xtrain, ytrain, xtest, ytest = data
                estimator = Lasso()
                estimator.fit(xtrain, ytrain)
                # ypred = estimator.predict(xtest[:, [it]])
                # try:
                #    f1mac = f1_score(ytest, ypred, average="macro")
                # except Exception as e3:
                #    excp.append(e3)
                # tmpscores[dk] = f1mac
                coef[f, :] = np.abs(estimator.coef_)

        rank = np.mean(coef, axis=0)
        return list(np.argsort(rank)[::-1]), rank

    def evaluate(s, cfg="default", clf="RF"):
        tmpscores = np.zeros(5, dtype=float)
        classifier = classifiers[clf]
        a = []
        b = []
        c = []
        d = []
        e = []
        f = []
        for dk, data in enumerate(ckf.data_splits()):
            xtrain, ytrain, xtest, ytest = data
            a.append(xtrain[:, s])
            b.append(ytrain)
            c.append(xtest[:, s])
            d.append(ytest)
            e.append(clf)
            f.append(cfg)

        with Pool(processes=params.njobs) as p:
            # 'results' is a List of Dict
            results = p.starmap(cv_fold, zip(e, f, a, b, c, d, [params] * 5))
            tmpscores = np.array(results)

        return np.mean(tmpscores)

    def evaluate2(s, cfg="default"):
        classifier = classifiers["RF"]
        estimator = classifier["model"](**classifier[cfg])
        scores = cross_validate(
            estimator,
            ckf.xdata[:, s],
            ckf.ydata,
            cv=ckf,
            n_jobs=5,
            scoring=["f1_macro", "accuracy"],
            error_score="raise",
        )
        return np.mean(scores["test_f1_macro"])

    def evaluate_hrw_4(subset, x, y, estimator, cv=None):
        #logger.debug(f'inner cross-validation hybrid ranking wrapper 4 folds')
        scores = cross_validate(
            estimator, 
            x[:, subset], 
            y, 
            cv=cv, 
            n_jobs=4, 
            scoring=["f1_macro", "accuracy"],
            error_score="raise",
        )
        #return scores
        return np.mean(scores['test_f1_macro'])


    for fold in range(5):

        #if fold!=4:
        #    continue

        logger.debug("")
        logger.debug(f"========== fold {fold} ============")

        kfinner = InnerWellFold(outerfold=fold, params=params)

        tmp = {}

        # =====================================================================
        # SFS ou SBS
        # =====================================================================
        sfs = None
        classif = None
        estimator = None
        estimator = classifier['model'](**classifier['default'])

        #kfoldinner = InnerWellFold(outerfold=kf, params=params)
        kfoldinner = InnerWellFold2(outerfold=fold)
        kfoldinner.xdata = kfinner.xdata
        kfoldinner.ydata = kfinner.ydata
        kfoldinner.splits = kfinner.splits

        logger.debug(f"ranking {params.ranking} ...")

        if params.ranking == "RF":
            features, rank = ranking_rf(kfinner.xdata.shape[1], None, None, kfoldinner)
        elif params.ranking == "ANOVA":
            features, rank = ranking_anova(kfinner.xdata, kfinner.ydata)
        elif params.ranking == "LASSO":
            features, rank = ranking_lasso()
        elif params.ranking == "MI":
            features, rank = ranking_mi(kfinner.xdata, kfinner.ydata)

        logger.debug(f"scores {str(rank)}")
        logger.debug(f"feature rank sorted best to worst")
        logger.debug(f"  {str(features)}")

        # features = sorted(n)
        factor = 0.99
        maxfeatures = int(len(features) * 0.2)
        subset = []
        bestvalue = -1
        bestsubset = []
        currentvalue = -1
        currentsubset = []

        # breakpoint()

        iter_ = 0
        data = []
        while len(features) > 0:
            iter_ = iter_ + 1
            # feat = features.pop(ranklist.pop(0))
            feat = features.pop(0)
            subset.append(feat)

            classifier = classifiers["RF"]
            estimator = classifier["model"](**classifier[cfg])

            #value = evaluate(subset, cfg=cfg, clf=clf)
            value = evaluate_hrw_4(subset, kfoldinner.xdata, kfoldinner.ydata, estimator, cv=kfoldinner)

            if value > currentvalue * factor:
                currentvalue = value
                currentsubset = subset[:]
                logger.debug(f"{iter_:3d} feature {feat:2d} melhorou")
                logger.debug(
                    f"{iter_:3d} current subset [{len(currentsubset):2d}, {currentvalue:.4f}] {str(currentsubset)}"
                )
            else:
                logger.debug(f"{iter_:3d} feature {feat:2d} nao melhorou, descartar")
                #currentsubset = subset[:-1]
                subset = subset[:-1]

            if currentvalue > bestvalue:
                bestvalue = currentvalue
                bestsubset = currentsubset[:]
            else:
                if len(currentsubset) > maxfeatures:
                    currentvalue = bestvalue
                    currentsubset = bestsubset

            #logger.debug(
            #    f"{iter_:3d} current subset [{len(currentsubset):2d}, {currentvalue:.4f}] {str(currentsubset)}"
            #)

            data.append({
                'algo': 'HRW',
                'fold': fold,
                'iter': iter_,
                'metric': currentvalue,
                'size': len(currentsubset),
                'try': feat,
                'ranking': params.ranking,
            })

        frame = pd.DataFrame(data=data)
        frame.to_excel(f"{params.experiment}_hybrid.xlsx")

        logger.debug(f"current subset [{len(currentsubset)}] {str(currentsubset)}")
        logger.debug(f"current value  {str(currentvalue)}")
        logger.debug(f'retrain default     {evaluate(currentsubset, "default", clf):.4f}')
        logger.debug(f'retrain best config {evaluate(currentsubset, "best", clf):.4f}')

        if len(currentsubset) > maxfeatures:
            ckf = CustomKFold(params=params, random_state=1)
            logger.debug(f"running SFS backward ({clf} {cfg})")
            estimator = classifier["model"](**classifier[cfg])
            sfs = SequentialFeatureSelector(
                estimator,
                n_features_to_select=maxfeatures,
                direction="backward",
                #cv=5,
                #cv=ckf,
                cv=kfoldinner,
                n_jobs=4,
                scoring="f1_macro",
            )

            #sfs.fit(ckf.xdata[:, bestsubset], ckf.ydata)
            sfs.fit(kfoldinner.xdata[:, bestsubset], kfoldinner.ydata)

            sfbs = sfs.get_support()

            newlist = []
            for c in range(len(bestsubset)):
                logger.debug(f"{c:2d} feature={bestsubset[c]:2d} SFB use={str(sfbs[c]):5s}  ")
                if sfbs[c]:
                    newlist.append(bestsubset[c])

            logger.debug("final refit after SBS")
            logger.debug(str(newlist))
            # scores = cross_validate(estimator, x_data[:, newlist], y_data, cv=5, n_jobs=5, scoring=['f1_macro', 'accuracy'], error_score='raise')
            # logger.debug(str(scores))
            logger.debug(f'retrain default     {evaluate(newlist, "default"):.4f}')
            logger.debug(f'retrain best config {evaluate(newlist, "best"):.4f}')
        
        else:
            newlist = bestsubset[:]
        
        logger.info('training with SELECTED features')
        estimator = None
        estimator = classifier['model'](**classifier['default'])

        #for f in wkf.split():
        for fd, fdata in enumerate(ckf.data_splits()):
            if fd == fold:
                estimator.fit(fdata[0][:, newlist], fdata[1])
                ypred = estimator.predict(fdata[2][:, newlist])

                try:
                    f1macro = f1_score(fdata[3], ypred, average="macro")
                except Exception as e3:
                    print(e3)
                    raise e3
                logger.info(f'final F1={f1macro:.4f}, fold {fold}, NoF={len(newlist)}')
        logger.info(f'testing SELECTED features FOLD {fold}')
        logger.info(f'new features {str(newlist)}')


    sys.exit(0)


def concat_excel(*files, output="compilado.xlsx"):
    lista = []
    for f in files:
        frame = pd.read_excel(f, header=0)
        lista.append(frame)
    final = pd.concat(lista, axis=0)
    # print(final.columns)
    final["case"] = list(map(lambda x: x[-2:-1], final["experiment"]))
    final["scenario"] = list(map(lambda x: x[-1], final["experiment"]))
    final["welltest"] = 0
    final.to_excel(output, index=False)


def ga_fs(round_, fold, params):
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter(params.logformat)
    fh = logging.FileHandler(f"{params.experiment}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    logger.info(params.name)
    logger.info("=" * 79)

    t0 = time.time()

    # genetic algorithm feature selection
    paramdict = vars(params)
    paramdict["obj"] = params
    paramdict["verbose"] = 2
    paramdict["dimension"] = 55
    paramdict["population"] = 100
    paramdict["max_generations"] = 200
    paramdict["mutation_rate"] = 0.01
    paramdict["crossover_rate"] = 0.99
    paramdict["best_to_keep"] = 2
    paramdict["xdata"] = None

    paramdict["classifiercfg"] = "default"
    # paramdict['classifiercfg'] = 'best'
    # breakpoint()

    garesult = ga_optimizer(objective1, paramdict)
    bestv = objective1(garesult["best_vector"], paramdict)

    frame = pd.DataFrame(data=garesult["generations"])
    frame.to_excel(f"ga_generations_{paramdict['experiment']}.xlsx")
    
    logger.debug(garesult)
    logger.debug(garesult["best_vector"])
    logger.debug(bestv)
    logger.debug(f"F1 Macro = {bestv['objective']:.4f}")


def ga_fs_inner(round_, fold, params, *args, **kwargs):
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter(params.logformat)
    fh = logging.FileHandler(f"{params.experiment}_ga.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    logger.info(params.name)
    logger.info("=" * 79)

    t0 = time.time()

    # genetic algorithm feature selection
    paramdict = vars(params)
    paramdict["obj"] = params
    paramdict["verbose"] = 2
    paramdict["dimension"] = 55
    paramdict["population"] = 100
    paramdict["max_generations"] = 200
    #paramdict["max_generations"] = 1
    paramdict["mutation_rate"] = 0.01
    paramdict["crossover_rate"] = 0.99
    paramdict["best_to_keep"] = 2
    paramdict["xdata"] = None
    paramdict["classifiercfg"] = "default"
    paramdict["outer_fold"] = None
    paramdict["njobsga"] = 6
    paramdict["njobscv"] = 1

    wkf = WellKFold(params=params, random_state=1)
    ckf = CustomKFold(params=params, random_state=1)

    clf = "RF"
    cfg = "default"
    n = len(params.usecols) * params.nfeaturesvar
    classifiers = get_classifiers(params.classifierstr)
    classifier = classifiers[clf]

    resultdata = []
    adata = []

    logger.debug(f"==========")
    logger.debug(f"{params.experiment} nested")
    logger.debug(f"==========")

    for fold in range(5):

        logger.debug("")
        logger.debug(f"========== fold {fold} ============")

        kfinner = InnerWellFold(outerfold=fold, params=params)

        tmp = {}

        # =====================================================================
        # SFS ou SBS
        # =====================================================================
        sfs = None
        classif = None
        estimator = None
        estimator = classifier['model'](**classifier['default'])

        #kfoldinner = InnerWellFold2(outerfold=fold, params=params)
        #kfoldinner.xdata = kfinner.xdata
        #kfoldinner.ydata = kfinner.ydata
        #kfoldinner.splits = kfinner.splits

        #for kf, data in enumerate(ckf.split()):
        #    xtr, ytr, xte, yte = data
        #    adata.append((xtr, ytr, xte, yte))

        #for kf, data in enumerate(adata):
        #    logger.debug(f'outer fold {kf}')
        #    paramdict['outer_fold'] = kf
        #    garesult = ga_optimizer(objective4, paramdict)
        #    bestv = objective4(garesult["best_vector"], paramdict)
        #    frame = pd.DataFrame(data=garesult["generations"])
        #    frame.to_excel(f"{paramdict['experiment']}_ga_generations_{kf}.xlsx")

        paramdict['outer_fold'] = fold
        paramdict['inner'] = kfinner
        #paramdict['inner'] = kfoldinner
        garesult = ga_optimizer(objective4, paramdict, logger)
        
        #bestv = objective4(garesult["best_vector"], paramdict)
        frame = pd.DataFrame(data=garesult["generations"])
        try:
            frame.to_excel(f"{params.experiment}_ga_generations_{fold}.xlsx")
        except Exception as e:
            logger.exception(e)

        newlist = garesult["best_vector"][:]
        r = np.where(newlist > 0)
        sfeatures = r[0]

        # logger.debug(garesult)
        logger.debug(garesult["best_vector"])
        #logger.debug(bestv)
        #logger.debug(f"F1 Macro = {bestv['objective']:.4f}")

        logger.info(f'training with SELECTED features for fold {fold}')
        estimator = None
        estimator = classifier['model'](**classifier['default'])

        #for f in wkf.split():
        for fd, fdata in enumerate(ckf.data_splits()):
            if fd == fold:
                estimator.fit(fdata[0][:, sfeatures], fdata[1])
                ypred = estimator.predict(fdata[2][:, sfeatures])

                try:
                    f1macro = f1_score(fdata[3], ypred, average="macro")
                except Exception as e3:
                    print(e3)
                    raise e3
                logger.info(f'final F1={f1macro:.4f}, fold {fold}, NoF={len(sfeatures)}, GA_time={garesult["time_elapsed"]:7.1f}s')
        logger.info(f'testing SELECTED features FOLD {fold}')

execution_dict = {
    "time_elapsed": 0,
    "algorithm": "",
    "metrics": {
        "best": [],
        "all_best": 0.0,
        "all_worst": 0.0,
        "time": [],
        "objective": [],
        "scores": [],
        "error_rate": [],
    },
    "n_eval": 0,
    "max_func_eval": 100,
}


def ga_optimizer(obj_func, parameters, logger):
    """
    
    Crossover: random swap
    Mutation: flip 1 element from 1% of normal distribution (gauss) population
    Elitism: keep 'k' best

    Inspired by:
    https://github.com/7ossam81/EvoloPy/blob/master/GA.py
    https://github.com/7ossam81/EvoloPy/blob/master/optimizers/GA.py

    """
    seed = parameters.get("seed", 1)
    lower = parameters.get("lower", [-1.0])
    upper = parameters.get("upper", [1.0])
    dimension = parameters.get("dimension", 30)
    population = parameters.get("population", 100)
    max_generations = parameters.get("max_generations", 100)
    mutation_rate = parameters.get("mutation_rate", 0.01)
    crossover_rate = parameters.get("crossover_rate", 1.0)
    crossover_alpha = parameters.get("crossover_alpha", 0.25)
    max_func_eval = parameters.get("max_func_eval", 10000)
    best_to_keep = parameters.get("best_to_keep", 2)
    verbose = parameters.get("verbose", 0)

    # np.random.seed(seed)
    # rnds = np.random.RandomState(seed)
    # rnds = np.random.default_rng(seed)
    rnds = np.random.Generator(np.random.PCG64(seed))
    
    # Input Objective function has variable dimensions
    # consider equi-distance "square" or "cube"
    if not isinstance(lower, list):
        lower = [lower]
    if not isinstance(upper, list):
        upper = [upper]

    lower = lower * dimension
    upper = upper * dimension

    exec_info = execution_dict.copy()
    n_eval = 0

    exec_info["metrics"]["time"] = np.zeros([max_generations], dtype="float")
    exec_info["metrics"]["best"] = np.zeros([max_generations, dimension], dtype="float")
    exec_info["metrics"]["scores"] = np.full(
        [max_generations, population], np.inf, dtype="float"
    )
    exec_info["metrics"]["error_rate"] = np.full(
        [max_generations, population], np.inf, dtype="float"
    )
    exec_info["metrics"]["objective"] = []
    exec_info["generations"] = []

    begin = time.time()

    # space = np.random.uniform(0, 1, [population, dimension]) * (upper[0] - lower[0]) + lower[0]
    # scores = np.random.uniform(0.0, 1.0, population) + 1000
    space = (
        rnds.uniform(0, 1, [population, dimension]) * (upper[0] - lower[0]) + lower[0]
    ).astype('float32')
    scores = rnds.uniform(0.0, 1.0, population) + 1000

    #
    def selection(space, scores):
        """
        """
        return space, scores

    # crossover takes 95% of running time!!
    def crossover(space, scores):
        search_space = np.zeros_like(space) + 1
        search_space[0:best_to_keep, :] = space[0:best_to_keep, :]

        for cross in range(best_to_keep + 1, population - 1, 2):

            # using SET parents will always be different
            # parents = set()

            # using LIST parents can repeat!
            parents = []

            parent1_idx = rnds.integers(low=0, high=population, size=1)
            parent2_idx = rnds.integers(low=0, high=population, size=1)

            parent1 = space[parent1_idx, :]
            parent2 = space[parent2_idx, :]

            child1 = np.zeros_like(parent1)
            child2 = np.zeros_like(parent2)

            # tmp = rnds.randint(0, 2, size=dimension)
            corte = rnds.integers(low=0, high=2, size=dimension)

            child1[:corte] = parent1[:corte]
            child1[corte:] = parent2[corte:]

            child2[:corte] = parent2[:corte]
            child2[corte:] = parent1[corte:]

            crossover_chance = rnds.uniform(0.0, 1.0)

            if crossover_chance < crossover_rate:
                search_space[cross, :] = np.copy(child1)
                search_space[cross + 1, :] = np.copy(child2)
            else:
                pass

        return search_space

    def crossover_3(space, scores):

        # search_space = np.zeros_like(space) + 1
        search_space = np.copy(space)
        # search_space[0:best_to_keep, :] = space[0:best_to_keep, :]
        crossover_chance = rnds.uniform(0.0, 1.0, size=population)

        for cross in range(best_to_keep + 1, population - 1, 2):

            parent1_idx = rnds.integers(low=0, high=population, size=1, endpoint=False)
            parent2_idx = rnds.integers(low=0, high=population, size=1, endpoint=False)

            parent1 = space[parent1_idx, :]
            parent2 = space[parent2_idx, :]

            child1 = np.zeros_like(parent1)
            child2 = np.zeros_like(parent2)

            corte = rnds.integers(low=0, high=dimension, size=None)

            child1[:corte] = parent1[:corte]
            child1[corte:] = parent2[corte:]

            child2[:corte] = parent2[:corte]
            child2[corte:] = parent1[corte:]

            if crossover_chance[cross] < crossover_rate:
                # print("*** crossover", cross, cross + 1, corte)
                search_space[cross, :] = np.copy(child1)
                search_space[cross + 1, :] = np.copy(child2)
            else:
                pass

        return search_space

    def crossover_prob(space, scores, prob):
        """
        Usar algum tipo de métrica de importância como distribuição de probabilidade
        """
        search_space = np.copy(space)
        crossover_chance = rnds.uniform(0.0, 1.0, size=population)

        for cross in range(best_to_keep + 1, population - 1, 1):
            # parent1_idx = rnds.integers(low=0, high=population, size=1, endpoint=True)
            # parent2_idx = rnds.integers(low=0, high=population, size=1, endpoint=True)

            # calling "CHOICE" two time in a row gives different results
            parent1_idx = rnds.choice(np.arange(population + 1), size=1, replace=False, p=prob, shuffle=False)
            parent2_idx = rnds.choice(np.arange(population + 1), size=1, replace=False, p=prob, shuffle=False)

            parent1 = space[parent1_idx, :]
            parent2 = space[parent2_idx, :]

            child = np.zeros_like(parent1)

            corte = rnds.integers(low=1, high=dimension, size=1)
            child[:corte] = parent1[:corte]
            child[corte:] = parent2[corte:]

            if crossover_chance[cross] < crossover_rate:
                search_space[cross, :] = np.copy(child)
            else:
                pass

        return search_space

    def crossover_cut(space, scores, coef):
        """
        Usar algum tipo de métrica de importância como parâmetro para o "corte"
        """
        search_space = np.copy(space)
        crossover_chance = rnds.uniform(0.0, 1.0, size=population)

        for cross in range(best_to_keep, population, 1):
            parent1_idx = rnds.integers(low=0, high=population, size=1, endpoint=True)
            parent2_idx = rnds.integers(low=0, high=population, size=1, endpoint=True)

            # calling "CHOICE" two time in a row gives different results
            # parent1_idx = rnds.choice(np.arange(population + 1), size=1, replace=False, p=prob, shuffle=False)
            # parent2_idx = rnds.choice(np.arange(population + 1), size=1, replace=False, p=prob, shuffle=False)

            parent1 = space[parent1_idx, :]
            parent2 = space[parent2_idx, :]

            child = np.zeros_like(parent1)

            sp1 = 0.0
            sp2 = 0.0
            for c in coef:
                pass

            cut = rnds.integers(low=1, high=dimension, size=1)
            child[:cut] = parent1[:cut]
            child[cut:] = parent2[cut:]

            if crossover_chance[cross] < crossover_rate:
                search_space[cross, :] = np.copy(child)
            else:
                pass

        return search_space

    def mutation(space, gen):
        n_mutate = np.int(population * mutation_rate)
        for m in range(n_mutate):
            # keep best => do not mutate
            #rand_individual = rnds.randint(best_to_keep + 1, population)
            rand_individual = rnds.integers(low=best_to_keep + 1, high=population, size=None)
            # decrease stdev with generations
            # stdev = 5.0 / np.sqrt(gen + 1)
            cromo = rnds.integers(low=0, high=dimension, size=None)

            # flip 1 cromossomo
            space[rand_individual, cromo] = 0 if space[rand_individual, cromo] else 1
            # print("*** mutate", rand_individual, cromo)

        return space

    def mutation_2(space, gen):
        n_mutate = np.int(population * mutation_rate)
        for m in range(n_mutate):
            rand_individual = rnds.randint(best_to_keep + 1, population)
            cromo = rnds.randint(0, dimension)
            space[rand_individual, cromo] = 0 if space[rand_individual, cromo] else 1
        return space

    def sort_iter(_space, _scores, errors):
        idx = scores.argsort()
        _space = _space[idx]
        _scores = _scores[idx]
        errors = errors[idx]
        return _space, _scores, errors

    def sort_iter2(_space, _scores):
        """
        Sort scores from minimum to maximum, that is, best to worst.
        """
        idx = scores.argsort()
        _space = _space[idx]
        _scores = _scores[idx]
        return _space, _scores

    def eval_obj(func, _space, _params) -> np.ndarray:
        _scores = np.full(population, np.inf)
        _errors = np.full(population, np.inf)

        # print(f"  eval objective function {population} times with {_params['njobs']} core(s)")

        if _params["njobsga"] > 1:
            logger.debug(f"  Eval individuals in parallel n={_params['njobsga']}")
            fspace = [_space[p, :] for p in range(population)]
            with Pool(processes=_params["njobsga"]) as p:
                # 'results' is a List of Dict
                results = p.starmap(func, zip(fspace, [_params] * population))
            for pk, r in enumerate(results):
                _scores[pk] = r["score"]
        else:
            for p in range(population):
                logger.debug(f'  Individual {p:03d}')
                r = func(_space[p, :], _params)
                _scores[p] = r["score"]
                if verbose > 1:
                    # print(f"individual {p:3d} => score {_scores[p]:.4f}  {str(_space[p, :])}")
                    pass

        # return _scores, _errors
        return _scores

    # init search space inside bounds
    for i in range(dimension):
        space[:, i] = rnds.uniform(0, 1, population) * (upper[i] - lower[i]) + lower[i]
        # space[:, i] = np.random.uniform(0, 1, population)

    # make binary
    space[space > 0.5] = 1
    space[space <= 0.5] = 0
    space = space.astype('int32')

    for _iter in range(0, max_generations):
        t1 = time.time()
        n_eval += population

        logger.info(f'Generation {_iter+1:04d}')

        space, scores = selection(space, scores)

        # crossover
        # space = crossover(space, scores)
        # space = crossover_1(space, scores)
        # space = crossover_2(space, scores)
        space = crossover_3(space, scores)

        # mutation
        space = mutation(space, _iter)

        # evaluate objective
        # nn_model, scores, error_rate = eval_obj(obj_func, space)
        scores = eval_obj(obj_func, space, parameters)

        if verbose > 1:
            logger.debug(f"best individual PREVIOUS GEN {np.argmin(scores):3d}")
            # pass

        # sort
        # space, scores, error_rate = sort_iter(space, scores, error_rate)
        space, scores = sort_iter2(space, scores)

        # if verbose > 1:
        #    print('best individual AFTER', np.argmin(scores))
        t2 = time.time()

        # save
        exec_info["metrics"]["scores"][_iter] = scores
        # exec_info['metrics']['error_rate'][_iter] = error_rate
        exec_info["metrics"]["time"][_iter] = t2 - begin
        exec_info["metrics"]["best"][_iter, :] = space[0, :]
        exec_info["metrics"]["objective"].append(scores[0])

        if verbose > 1:
            logger.debug(f"best individual CURRENT GEN {np.argmin(scores)}")
            logger.debug('top 5')
            for i in range(0, 5):
                #print(f"{str(space[i, :])} score = {scores[i]:.4f}")
                logger.debug(f"  score = {scores[i]:.4f}")

        if verbose and _iter % np.fmax(np.int((max_generations + 1) / 10),1) == 0:
            logger.debug(
                f"    finished gen {_iter:04d}, "
                f"f_min = {scores[0]:+13.5e}, "
                f"time = {time.time() - begin:8.3f}s "
                "nof="
                f"{np.sum(space[0, :])}"
            )

        exec_info["generations"].append(
            {
                "elapsed_time": exec_info["metrics"]["time"][_iter],
                "current_time": t2 - t1,
                "best_vector": str(space[0, :].astype(int)),
                "score": scores[0],
                "objective": 1 - scores[0],
                "positive": np.sum(space[0, :]),
                "negative": len(space[0, :]) - np.sum(space[0, :]),
            }
        )

        # if n_eval >= max_func_eval:
        #    print(f'\t\t   stopped max eval {n_eval}')
        #    break

    if verbose:
        logger.debug(
            f"\t\tIteration {_iter:05d} (all generations), "
            f"f_min = {scores[0]:+13.5e}, "
            f"time = {time.time() - begin:8.3f}s "
            "nof="
            f"{np.sum(space[0, :])}"
        )
        # print('5 best feature groups:')
        # print(space[:5, :])
        logger.debug("5 best scores (minimum error)")
        logger.debug(scores[:5])

    exec_info["algorithm"] = "GA - Genetic Algorithm"
    exec_info["time_elapsed"] = time.time() - begin
    exec_info["n_eval"] = n_eval
    exec_info["metrics"]["all_best"] = scores[0]
    exec_info["best_individual"] = space[0, :]
    exec_info["best_score"] = scores[0]
    exec_info["best_vector"] = space[0, :]

    return exec_info


def pso_optimizer(obj_func, parameters):
    """

    Inspired by:
    https://github.com/7ossam81/EvoloPy/blob/master/PSO.py

    """
    lower = parameters.get("lower", [-1.0])
    upper = parameters.get("upper", [1.0])
    dimension = parameters.get("dimension", 30)
    population = parameters.get("population", 100)
    seed = parameters.get("seed", 1)
    max_iterations = parameters.get("max_iterations", 100)
    velocity_max = np.full(dimension, parameters.get("velocity_max", 5.0))
    max_func_eval = parameters.get("max_func_eval", 100)
    verbose = parameters.get("verbose", 0)

    np.random.seed(seed)

    w_max = parameters.get("w_max", 0.9)
    w_min = parameters.get("w_min", 0.2)

    if not isinstance(lower, list):
        lower = [lower]
    if not isinstance(upper, list):
        upper = [upper]

    lower = lower * dimension
    upper = upper * dimension

    # track execution
    exec_info = execution_dict.copy()
    exec_info["metrics"]["scores"] = np.zeros(max_iterations, dtype="float")
    exec_info["metrics"]["objective"] = []  # np.zeros(max_iterations, dtype='float')

    # inner variables

    c1 = 2.0
    c2 = 2.0

    velocity = np.full([population, dimension], 0.0)

    p_best_score = np.full(population, np.inf)
    p_best = np.full([population, dimension], 0.0)

    g_best_score = np.inf
    g_best = np.full(population, 0.0)

    space = np.full([population, dimension], 0.0)

    begin = time.time()

    # init fill
    for _d in range(dimension):
        rand_dim = np.random.uniform(0, 1, population)
        space[:, _d] = rand_dim * (upper[_d] - lower[_d]) + lower[_d]

    n_eval = 0
    for _iter in range(max_iterations):

        _scores = np.full(population, np.inf)
        _errors = np.full(population, np.inf)

        if _params["njobs"] > 1:
            fspace = [_space[p, :] for p in range(population)]
            with Pool(processes=_params["njobs"]) as proc:
                # 'results' is a List of Dict
                results = proc.starmap(obj_func, zip(fspace, [_params] * population))
            for pk, r in enumerate(results):
                _scores[pk] = r["score"]
        else:
            for _p in range(population):
                r = obj_func(_space[_p, :], _params)
                _scores[_p] = r["score"]

        for _p in range(population):
            # keep inside bounds / search limits
            space[_p, :] = np.clip(space[_p, :], lower, upper)

            # eval
            # rdict = obj_func(space[_p, :], parameters)
            # score = rdict['score']
            score = _scores[_p]

            n_eval += 1

            # check
            if p_best_score[_p] > score:
                p_best_score[_p] = score
                p_best[_p, :] = np.copy(space[_p, :])

            if g_best_score > score:
                g_best_score = score
                g_best = np.copy(space[_p, :])

        w = w_max - _iter * ((w_max - w_min) / max_iterations)

        for _p in range(population):
            r1 = np.random.random([dimension])
            r2 = np.random.random([dimension])

            v = velocity[_p, :]

            a1 = c1 * r1 * (p_best[_p, :] - space[_p, :])
            a2 = c2 * r2 * (g_best - space[_p, :])

            velocity[_p, :] = w * v + a1 + a2

            velocity[_p, :] = np.fmax(velocity[_p, :], -velocity_max)  # negative!!
            velocity[_p, :] = np.fmin(velocity[_p, :], velocity_max)

            # UPDATE space
            space[_p, :] += velocity[_p, :]

        # save data to analyse and plot later
        exec_info["metrics"]["scores"][_iter] = g_best_score
        # exec_info['metrics']['objective'][_iter] = g_best_score
        exec_info["metrics"]["objective"].append(g_best_score)

        # some logging
        if verbose and _iter % np.int((max_iterations + 1) / 5.0) == 0:
            print(
                f"\t\tIteration {_iter:05d},",
                f"f_min = {g_best_score:+13.5e},",
                f"x = {p_best[0, 0]:+13.5e}",
                f"n_eval = {n_eval:05d}",
                f"time = {time.time() - begin:8.3f}s",
            )

        if n_eval >= max_func_eval:
            print(f"\t\t   stopped max eval {n_eval}")
            break

    # finished
    exec_info["algorithm"] = "PSO - Particle Swarm Optimization"
    exec_info["time_elapsed"] = time.time() - begin
    exec_info["n_eval"] = n_eval
    exec_info["metrics"]["all_best"] = np.amin(exec_info["metrics"]["objective"])
    exec_info["best_particle"] = g_best
    exec_info["best_score"] = g_best_score
    exec_info["best_vector"] = g_best  # nomenclatura para padronizar o retorno

    return exec_info


def objective1(binfeatures, params_) -> Dict:
    """
    Objective function for Genetic Algorithm optimization (GA)
    """
    params = dict(xdata=None, classifiercfg="default", classifierkey="RF",)
    params.update(params_)
    tmpscores = np.zeros(5, dtype=float)
    classifiers = get_classifiers(params["classifierstr"])
    classifier = classifiers[params["classifierkey"]]

    r = np.where(binfeatures > 0)
    features = r[0]

    # print(f'objective1 CV=5 with {len(features)} features')
    # print(f'objective1 CV=5 with [{len(features)}] {str(features)}')

    excp = []
    round_ = 1

    for fold in range(params["nfolds"]):
        win = params["windowsize"]
        step = params["stepsize"]
        case = params["experiment"][-2:]
        with h5py.File(f"datasets_folds_exp{case}.h5", "r") as ffolds:

            group = "pos"
            gk = f"/case{case}_{group}_r{round_}_nf{params['nfolds']}_f{fold}_w{win}_s{step}"
            trainpositive = ffolds[gk]["xtrain"][()]
            testpositive = ffolds[gk]["xvalid"][()]

            group = "neg"
            gk = f"/case{case}_{group}_r{round_}_nf{params['nfolds']}_f{fold}_w{win}_s{step}"
            trainnegative = ffolds[gk]["xtrain"][()]
            testnegative = ffolds[gk]["xvalid"][()]

            gk = f"/case{case}_{group}_r{round_}_nf{params['nfolds']}_f-test_w{win}_s{step}"

        xtrain, ytrain = vertical_split_bin(trainnegative, trainpositive)
        xtest, ytest = vertical_split_bin(testnegative, testpositive)

        mask = np.any(
            np.isnan(xtrain)
            | (xtrain > np.finfo(np.float32).max)
            | np.isinf(xtrain)
            | ~np.isfinite(xtrain),
            axis=1,
        )

        xtrain = xtrain[~mask]
        ytrain = ytrain[~mask]

        mask = np.any(
            np.isnan(xtest)
            | (xtest > np.finfo(np.float32).max)
            | np.isinf(xtest)
            | ~np.isfinite(xtest),
            axis=1,
        )

        xtest = xtest[~mask]
        ytest = ytest[~mask]

        scalerafter = StandardScaler()
        scalerafter.fit(xtrain)
        xtrain = scalerafter.transform(xtrain)
        xtest = scalerafter.transform(xtest)

        estimator = classifier["model"](**classifier[params["classifiercfg"]])
        estimator.fit(xtrain[:, features], ytrain)
        ypred = estimator.predict(xtest[:, features])

        try:
            f1mac = f1_score(ytest, ypred, average="macro")
        except Exception as e3:
            excp.append(e3)

        tmpscores[fold] = f1mac

    objective = np.mean(tmpscores)

    return {
        "score": 1.0 - objective,  # search for the minimum!!
        "objective": objective,
        "errors": excp,
    }


def objective2(binfeatures, params_) -> Dict:
    """
    Objective function for Particle Swarm Optimization (PSO)
    """
    params = dict(xdata=None, classifiercfg="default", classifierkey="RF",)
    params.update(params_)
    tmpscores = np.zeros(5, dtype=float)
    classifiers = get_classifiers(params["classifierstr"])
    classifier = classifiers[params["classifierkey"]]

    r = np.where(binfeatures > 0)
    features = r[0]

    # print(f'objective1 CV=5 with {len(features)} features')
    # print(f'objective1 CV=5 with [{len(features)}] {str(features)}')

    excp = []
    round_ = 1

    for fold in range(params["nfolds"]):
        win = params["windowsize"]
        step = params["stepsize"]
        case = params["experiment"][-2:]
        with h5py.File(f"datasets_folds_exp{case}.h5", "r") as ffolds:

            group = "pos"
            gk = f"/case{case}_{group}_r{round_}_nf{params['nfolds']}_f{fold}_w{win}_s{step}"
            trainpositive = ffolds[gk]["xtrain"][()]
            testpositive = ffolds[gk]["xvalid"][()]

            group = "neg"
            gk = f"/case{case}_{group}_r{round_}_nf{params['nfolds']}_f{fold}_w{win}_s{step}"
            trainnegative = ffolds[gk]["xtrain"][()]
            testnegative = ffolds[gk]["xvalid"][()]

            gk = f"/case{case}_{group}_r{round_}_nf{params['nfolds']}_f-test_w{win}_s{step}"

        xtrain, ytrain = vertical_split_bin(trainnegative, trainpositive)
        xtest, ytest = vertical_split_bin(testnegative, testpositive)

        mask = np.any(
            np.isnan(xtrain)
            | (xtrain > np.finfo(np.float32).max)
            | np.isinf(xtrain)
            | ~np.isfinite(xtrain),
            axis=1,
        )

        xtrain = xtrain[~mask]
        ytrain = ytrain[~mask]

        mask = np.any(
            np.isnan(xtest)
            | (xtest > np.finfo(np.float32).max)
            | np.isinf(xtest)
            | ~np.isfinite(xtest),
            axis=1,
        )

        xtest = xtest[~mask]
        ytest = ytest[~mask]

        scalerafter = StandardScaler()
        scalerafter.fit(xtrain)
        xtrain = scalerafter.transform(xtrain)
        xtest = scalerafter.transform(xtest)

        estimator = classifier["model"](**classifier[params["classifiercfg"]])
        estimator.fit(xtrain[:, features], ytrain)
        ypred = estimator.predict(xtest[:, features])

        try:
            f1mac = f1_score(ytest, ypred, average="macro")
        except Exception as e3:
            excp.append(e3)
        tmpscores[fold] = f1mac

    objective = np.mean(tmpscores)

    return {
        "score": 1.0 - objective,  # search for the minimum!!
        "objective": objective,
        "errors": excp,
    }


def objective3(features, params_):
    params = dict(xdata=None, classifiercfg="default", classifierkey="RF",)
    params.update(params_)
    
    classifiers = get_classifiers(params["classifierstr"])
    classifier = classifiers[clf]
    estimator = classifier["model"](**classifier[cfg])

    ckf = CustomKFold(params=params_, random_state=1)

    scores = cross_validate(
        estimator,
        ckf.xdata[:, features],
        ckf.ydata,
        cv=ckf,
        n_jobs=params_['njobs'],
        scoring=["f1_macro", "accuracy"],
        error_score="raise",
    )

    objective = np.mean(scores["test_f1_macro"])

    return {
        "score": 1.0 - objective,  # search for the minimum!!
        "objective": objective,
        "errors": None,
    }


def objective4(binfeatures, params_):
    """
    """
    params = dict(xdata=None, classifiercfg="default", classifierkey="RF",)
    params.update(params_)
    clf = "RF"
    cfg = 'default'
    classifiers = get_classifiers(params["classifierstr"])
    classifier = classifiers[clf]
    estimator = classifier["model"](**classifier[cfg])
    r = np.where(binfeatures > 0)
    features = r[0]

    #outer = WellKFold(params=params, random_state=1)
    #inner = InnerWellFold(outerfold=params['outer_fold'], params=params['obj'])
    inner = params['inner']
    errors = []
    objective = 0.0
    scores = {}

    try:
        scores = cross_validate(
            estimator,
            #inner.xdata[:, binfeatures],
            inner.xdata[:, features],
            inner.ydata,
            cv=inner,
            #n_jobs=params_['njobs'],
            n_jobs=1,
            scoring=["f1_macro", "accuracy"],
            error_score="raise",
        )

        objective = np.mean(scores["test_f1_macro"])

    except Exception as e:
        breakpoint()
        errors.append(e)
        #logger.exception(e)
        print(e)
    
    print(f'    GA {objective:.4f} {len(r[0]):2d} {str(r[0])}')

    return {
        "score": 1.0 - objective,  # search for the minimum!!
        "objective": objective,
        "features": features,
        "scores": scores,
        "errors": errors,
        "outer_fold_ga": params['outer_fold'],
    }


if __name__ == "__main__":
    # os.system('cls')
    # os.system("clear")
    fire.Fire(
        {
            "concatexcel": concat_excel,
            "csv2hdf": csv2hdf,
            "csv2hdfpar": csv2hdfpar,
            "cleandataset": cleandataset,
            "cleandataseth5": cleandataseth5,
        }
    )
