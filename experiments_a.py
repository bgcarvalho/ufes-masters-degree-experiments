from typing import Tuple
#from dataclasses import dataclass
import fire
import h5py
from experiments import (
    cleandataseth5,
    runexperiment,
    cleandataset,
    csv2hdf,
    DefaultParams,
    split_and_save3,
    singlefold,
    hybrid_ranking_wrapper,
    hybrid_ranking_wrapper_nested,
    ga_fs_inner,
    fs5foldcv,
)


#@dataclass(frozen=False)
class Params(DefaultParams):
    """
    Experiment configuration (and model hyperparameters)

    DataClass offers a little advantage over simple dictionary in that it checks if
    the parameter actually exists. A dict would accept anything.
    """

    name = "Experiment 03A"
    experiment: str = "experiment3a"
    shuffle: bool = True
    # shuffle: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, i in kwargs.items():
            setattr(self, k, i)


    def __post_init__(self):
        super().__post_init__()
        self.positive = [4]
        self.negative = [0]


def read_and_split_h5(fold: int, round_: int, params: Params) -> Tuple:
    """
    HDF files offer at least a couple advantages:
    1 - reading is faster than CSV
    2 - you dont have to read the whole dataset to get its size (shape)

    H5PY fancy indexing is very slow.
    https://github.com/h5py/h5py/issues/413
    """
    win = params.windowsize
    step = params.stepsize
    case = params.experiment[-2:]
    with h5py.File(f"datasets_folds_exp{case}.h5", "r") as ffolds:

        group = "pos"
        gk = f"/case{case}_{group}_r{round_}_nf{params.nfolds}_f{fold}_w{win}_s{step}"
        trainpositive = ffolds[gk]["xtrain"][()]
        testpositive = ffolds[gk]["xvalid"][()]

        group = "neg"
        gk = f"/case{case}_{group}_r{round_}_nf{params.nfolds}_f{fold}_w{win}_s{step}"
        trainnegative = ffolds[gk]["xtrain"][()]
        testnegative = ffolds[gk]["xvalid"][()]

        gk = f"/case{case}_{group}_r{round_}_nf{params.nfolds}_f-test_w{win}_s{step}"

    return trainnegative, trainpositive, testnegative, testpositive, None, None


def run_experiment(*args, **kwargs):
    params = Params(**kwargs)
    # params.read_and_split = read_and_split_bin
    params.read_and_split = read_and_split_h5
    return runexperiment(params, *args, **kwargs)


def feature(*args, **kwargs):
    params = Params(**kwargs)
    params.read_and_split = read_and_split_h5
    return singlefold(1, 0, params, *args, **kwargs)


def ga_inner(*args, **kwargs):
    params = Params(**kwargs)
    params.read_and_split = read_and_split_h5
    return ga_fs_inner(1, 0, params, *args, **kwargs)

def hybrid(*args, **kwargs):
    params = Params(**kwargs)
    params.read_and_split = read_and_split_h5
    # return hybrid_ranking_wrapper(params, *args, **kwargs)
    #return hybrid_ranking_wrapper(params)
    return hybrid_ranking_wrapper_nested(params)


def featurecv(*args, **kwargs):
    params = Params(**kwargs)
    params.read_and_split = read_and_split_h5
    return fs5foldcv(1, 0, params)

def split_folds_wells(*args, **kwargs):
    params = Params(**kwargs)
    split_and_save3(params, "3a", "pos", params.positive)
    split_and_save3(params, "3a", "neg", params.negative)

def gridsearch(*args, **kwargs):
    pass

def bayesscv(*args, **kwargs):
    params = Params(**kwargs)
    params.gridsearch = 2
    params.read_and_split = read_and_split_h5
    return runexperiment(params, *args, **kwargs)

if __name__ == "__main__":
    fire.Fire(
        {
            "runexperiment": run_experiment,
            "csv2hdf": csv2hdf,
            #"cleandataset": cleandataset,
            "cleandataset": cleandataseth5,
            "splitfolds": split_folds_wells,
            "gridsearch": gridsearch,
            "bayesscv": bayesscv,
            "feature": feature,
            "featurecv": featurecv,
            "hybrid": hybrid,
            "ga": ga_inner,
        }
    )
