import os.path as osp
import warnings
import sys

SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))

import pickle as pkl

from input_generator.prior_gen import PriorBuilder
from input_generator.utils import get_output_tag

from jsonargparse import CLI
from typing import List, Optional

def merge_statistics(
    save_dir: str,
    prior_tag: str,
    prior_builders: List[PriorBuilder],
    names: List[str],
    tag: Optional[str] = None
):
    """
    Merges statistics computed for separate datasets or for individual samples of the same dataset.

    Parameters
    ----------
    save_dir : str
        Path to directory in which output will be saved
    names : List[str]
        List of either sample names or dataset names for which statistics will be merged
    prior_tag : str
        String identifying the specific combination of prior terms
    prior_builders : List[PriorBuilder]
        List of PriorBuilder objects and their corresponding parameters
    tag : str
        Optional label included to specify dataset for which sample statistics will be merged
    """
    all_stats = []
    for name in names:
        stats_fn = osp.join(save_dir, f"{get_output_tag([tag, name, prior_tag], placement='before')}prior_builders.pck")
        if osp.exists(stats_fn):
            with open(stats_fn, "rb") as ifile:
                stats = pkl.load(ifile)
            all_stats.append(stats)
        else:
            warnings.warn(f"Sample {name} has no saved statistics - This entry will be skipped")
            continue
    
    fnout =  osp.join(save_dir, f"{get_output_tag([tag, prior_tag], placement='before')}prior_builders.pck")
    builder_dict = {}
    for prior_builder in prior_builders:
        builder_dict[prior_builder.name] = prior_builder

    for statistics in all_stats:
        for builder in statistics:
            combined_builder = builder_dict[builder.name]
            for nl_name in list(builder.histograms.data.keys()):
                if nl_name not in builder.nl_builder.nl_names:
                    continue
                hists = builder.histograms[nl_name]
                for k, hist in hists.items():
                    combined_builder.histograms.data[nl_name][k] += hist
    
    with open(fnout, "wb") as ofile:
        pkl.dump(prior_builders, ofile)


if __name__ == "__main__":
    CLI([merge_statistics], as_positional=False)
