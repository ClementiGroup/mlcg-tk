import os.path as osp
import sys

SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))

from input_generator.raw_dataset import RawDataset
from input_generator.embedding_maps import CGEmbeddingMap
from input_generator.prior_gen import PriorBuilder
from input_generator.prior_fit import HistogramsNL
from input_generator.prior_fit.fit_potentials import fit_potentials
from input_generator.utils import get_output_tag
from tqdm import tqdm
import torch
from time import ctime
import numpy as np
import pickle as pck
from typing import Dict, List, Union, Callable, Optional
from jsonargparse import CLI
from scipy.integrate import trapezoid
from collections import defaultdict
from copy import deepcopy
import warnings

# import seaborn as sns

from mlcg.nn.gradients import SumOut
from mlcg.utils import makedirs


def compute_statistics(
    dataset_name: str,
    names: List[str],
    tag: str,
    save_dir: str,
    stride: int,
    batch_size: int,
    prior_tag: str,
    prior_builders: List[PriorBuilder],
    embedding_map: CGEmbeddingMap,
    statistics_tag: Optional[str] = None,
    device: str = "cpu",
    save_figs: bool = True,
    save_sample_statistics: bool = False,
    weights_template_fn: Optional[str] = None,
):
    """
    Computes structural features and accumulates statistics on dataset samples

    Parameters
    ----------
    dataset_name : str
        Name given to specific dataset
    names : List[str]
        List of sample names
    tag : str
        Label given to all output files produced from dataset
    save_dir : str
        Path to directory from which input will be loaded and to which output will be saved
    stride: int
        Integer by which to stride frames
    batch_size : int
        Number of frames to take per batch
    prior_tag : str
        String identifying the specific combination of prior terms
    prior_builders : List[PriorBuilder]
        List of PriorBuilder objects and their corresponding parameters
    embedding_map : CGEmbeddingMap
        Mapping object
    statistics_tag : str
        String differentiating parameters used for statistics computation
    device: str
        Device on which to run delta force calculations
    save_sample_statistics:
        If true, will save individual list of prior builders with accumulated statistics of one molecule
    save_figs: bool
        Whether to plot histograms of computed statistics
    weights_template_fn : str
        Template file location of weights to use for accumulating statistics
    """

    all_nl_names = set()
    nl_name2prior_builder = {}
    for prior_builder in prior_builders:
        for nl_name in prior_builder.nl_builder.nl_names:
            all_nl_names.add(nl_name)
            nl_name2prior_builder[nl_name] = prior_builder

    dataset = RawDataset(dataset_name, names, tag)
    for samples in tqdm(
        dataset, f"Compute histograms of CG data for {dataset_name} dataset..."
    ):
        if not samples.has_saved_cg_output(save_dir, prior_tag):
            warnings.warn(f"Sample {samples.name} has no saved CG output - This entry will be skipped")
            continue
        if weights_template_fn != None and not osp.exists(
                osp.join(save_dir, weights_template_fn.format(samples.name))
                ):
            warnings.warn(f"Could not find weights for sample {samples.name}; the file {osp.join(save_dir, weights_template_fn.format(samples.name))} does not exist. This entry will be skipped.")
            continue

        batch_list = samples.load_cg_output_into_batches(
            save_dir, prior_tag, batch_size, stride, weights_template_fn=weights_template_fn,
        )
        nl_names = set(batch_list[0].neighbor_list.keys())

        assert nl_names.issubset(
            all_nl_names
        ), f"some of the NL names '{nl_names}' in {dataset_name}:{samples.name} have not been registered in the nl_builder '{all_nl_names}'"

        if save_sample_statistics:
            sample_fnout = osp.join(save_dir, f"{get_output_tag([samples.tag, samples.name, prior_tag, statistics_tag], placement='before')}prior_builders.pck")
            sample_prior_builders = [
                deepcopy(prior_builder) for prior_builder in prior_builders
            ]

            sample_nl_name2prior_builder = {}
            for prior_builder in sample_prior_builders: 
                for nl_name in prior_builder.nl_builder.nl_names:
                    if nl_name in prior_builder.histograms.data.keys() and nl_name not in nl_names: 
                        prior_builder.histograms.data.pop(nl_name)
                    prior_builder.histograms.data[nl_name].clear()
                    sample_nl_name2prior_builder[nl_name] = prior_builder

            for batch in tqdm(batch_list, f"molecule name: {samples.name}", leave=False):
                batch = batch.to(device)
                for nl_name in nl_names:
                    prior_builder = sample_nl_name2prior_builder[nl_name]
                    prior_builder.accumulate_statistics(nl_name, batch)
            
            with open(sample_fnout, "wb") as f:
                pck.dump(sample_prior_builders, f)
            
            continue # does not save accumulated statistics if sample statistics saved

        for batch in tqdm(batch_list, f"molecule name: {samples.name}", leave=False):
            batch = batch.to(device)
            for nl_name in nl_names:
                prior_builder = nl_name2prior_builder[nl_name]
                prior_builder.accumulate_statistics(nl_name, batch)

    key_map = {v: k for k, v in embedding_map.items()}
    if save_figs:
        for prior_builder in prior_builders:
            figs = prior_builder.histograms.plot_histograms(key_map)
            for tag, fig in figs:
                makedirs(osp.join(save_dir, f"{prior_tag}_plots"))
                fig.savefig(
                    osp.join(save_dir, f"{prior_tag}_plots", f"hist_{tag}.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
    
    if not save_sample_statistics:
        # cummulative statistics are only saved if individual statistics were not saved
        fnout = osp.join(save_dir, f"{get_output_tag([samples.tag, prior_tag], placement='before')}prior_builders.pck")
        with open(fnout, "wb") as f:
            pck.dump(prior_builders, f)


def fit_priors(
    save_dir: str,
    prior_tag: str,
    embedding_map: CGEmbeddingMap,
    temperature: float,
):
    """
    Fits potential energy estimates to computed statistics

    Parameters
    ----------
    save_dir : str
        Path to directory from which input will be loaded and to which output will be saved
    prior_tag : str
        String identifying the specific combination of prior terms
    embedding_map : CGEmbeddingMap
        Mapping object
    temperature : float
        Temperature from which beta value will be computed
    """
    prior_fn = osp.join(save_dir, f"{prior_tag}_prior_builders.pck")
    fnout = osp.join(save_dir, f"{prior_tag}_prior_model.pt")

    with open(prior_fn, "rb") as f:
        prior_builders = pck.load(f)

    nl_names = []
    nl_name2prior_builder = {}
    for prior_builder in prior_builders:
        for nl_name in list(prior_builder.histograms.data.keys()):
            nl_names.append(nl_name)
            nl_name2prior_builder[nl_name] = prior_builder
    prior_models = {}
    for nl_name in nl_names:
        prior_builder = nl_name2prior_builder[nl_name]
        prior_model = fit_potentials(
            nl_name=nl_name,
            prior_builder=prior_builder,
            embedding_map=embedding_map,
            temperature=temperature,
        )
        prior_models[nl_name] = prior_model

    modules = torch.nn.ModuleDict(prior_models)
    full_prior_model = SumOut(modules, targets=["energy", "forces"])
    torch.save(full_prior_model, fnout)


if __name__ == "__main__":
    print("Start fit_priors.py: {}".format(ctime()))

    CLI([compute_statistics, fit_priors])

    print("Finish fit_priors.py: {}".format(ctime()))
