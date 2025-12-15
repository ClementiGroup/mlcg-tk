import torch
from mlcg.geometry._symmetrize import _symmetrise_map


def neg_log_likelihood(y, yhat):
    """
    Convert dG to probability and use KL divergence to get difference between
    predicted and actual
    """
    L = torch.sum(torch.exp(-y) * torch.log(torch.exp(-yhat)))
    return -L


def compute_aic(real_distibution, ref_distribution, free_parameters):
    """Method for computing the AIC"""
    aic = (
        2
        * neg_log_likelihood(
            ref_distribution,
            real_distibution,
        )
        + 2 * free_parameters
    )
    return aic


def compute_nl_unique_keys(
    atom_types: torch.Tensor,
    mapping: torch.Tensor,
    device: str = "cpu",
):
    """
    Computes and returns the unique atom-type interaction keys for a given neighbor list.
    This runs only once per nl_name.
    """

    order = mapping.shape[0]

    interaction_types = torch.stack(
        [atom_types[mapping[ii]] for ii in range(order)], dim=0
    )
    interaction_types_sym = _symmetrise_map[order](interaction_types)
    if interaction_types_sym.numel() >0 :
        max_type = interaction_types_sym.max().item() + 1
        multipliers = torch.tensor(
            [max_type**i for i in range(order)],
            dtype=torch.long,
            device=interaction_types_sym.device,
        )
        hashed = (interaction_types_sym.long() * multipliers.unsqueeze(1)).sum(dim=0)

        unique_hashes, inverse_indices = torch.unique(hashed, return_inverse=True)
        n_unique = len(unique_hashes)

        first_occurrences = torch.zeros(n_unique, dtype=torch.long, device=hashed.device)
        for i, h in enumerate(unique_hashes):
            first_occurrences[i] = (hashed == h).nonzero(as_tuple=True)[0][0]
        unique_keys_in_data = interaction_types_sym[:, first_occurrences]

        nl_keys_dict = {
            "order": order,
            "unique_keys_in_data": unique_keys_in_data,
            "inverse_indices": inverse_indices,
        }
    else:
        nl_keys_dict = {
            "order": order,
            "unique_keys_in_data": torch.tensor([]),
            "inverse_indices": torch.tensor([]),
        }

    return nl_keys_dict
