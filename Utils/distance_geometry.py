import torch
from torch.optim import LBFGS


def complete_distance_matrix(edge_distances, atom_count):
    if atom_count < 2:
        raise ValueError("MDS reconstruction requires at least two atoms")
    expected = atom_count * (atom_count - 1)
    if edge_distances.numel() != expected:
        raise ValueError(
            f"Expected {expected} directed edge distances for {atom_count} atoms, "
            f"got {edge_distances.numel()}"
        )

    mask = ~torch.eye(atom_count, dtype=torch.bool, device=edge_distances.device)
    matrix = edge_distances.new_zeros((atom_count, atom_count))
    matrix[mask] = edge_distances.reshape(-1)
    matrix = 0.5 * (matrix + matrix.transpose(0, 1))
    matrix = matrix.clamp_min(0)
    matrix.fill_diagonal_(0)
    return matrix


def classical_mds(distance_matrix, dimensions=3):
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")
    if dimensions <= 0:
        raise ValueError("dimensions must be greater than zero")

    atom_count = distance_matrix.shape[0]
    distances = 0.5 * (distance_matrix + distance_matrix.transpose(0, 1))
    distances = distances.clamp_min(0).clone()
    distances.fill_diagonal_(0)

    identity = torch.eye(atom_count, dtype=distances.dtype, device=distances.device)
    centering = identity - torch.ones_like(distances) / atom_count
    gram = -0.5 * centering @ distances.square() @ centering
    gram = 0.5 * (gram + gram.transpose(0, 1))

    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    retained = min(dimensions, atom_count)
    eigenvalues = eigenvalues[-retained:].clamp_min(0)
    eigenvectors = eigenvectors[:, -retained:]
    coordinates = eigenvectors * eigenvalues.sqrt().unsqueeze(0)

    if retained < dimensions:
        padding = coordinates.new_zeros((atom_count, dimensions - retained))
        coordinates = torch.cat((coordinates, padding), dim=1)
    return coordinates


def align_to_reference(coordinates, reference):
    if coordinates.shape != reference.shape:
        raise ValueError("coordinates and reference must have the same shape")

    source_center = coordinates.mean(dim=0, keepdim=True)
    target_center = reference.mean(dim=0, keepdim=True)
    source = coordinates - source_center
    target = reference - target_center
    u, _, vh = torch.linalg.svd(source.transpose(0, 1) @ target)
    rotation = u @ vh
    return source @ rotation + target_center


def weighted_distance_loss(coordinates, target_distances, src, dst, epsilon=1e-6):
    reconstructed = torch.linalg.vector_norm(
        coordinates[src] - coordinates[dst], dim=-1
    )
    weights = (target_distances + epsilon).reciprocal().square()
    return ((reconstructed - target_distances).square() * weights).sum()


def pairwise_dist_to_coord_mds(
    reactant_pos,
    product_pos,
    pairwise_distance_ts_pred,
    max_iter=100,
    lr=0.1,
    epsilon=1e-6,
):
    if reactant_pos.shape != product_pos.shape or reactant_pos.ndim != 2:
        raise ValueError("reactant_pos and product_pos must have matching [N, D] shapes")
    if reactant_pos.shape[1] != 3:
        raise ValueError("MDS reconstruction currently requires three-dimensional inputs")
    if max_iter <= 0:
        raise ValueError("max_iter must be greater than zero")
    if lr <= 0:
        raise ValueError("lr must be greater than zero")

    atom_count = reactant_pos.shape[0]
    distance_matrix = complete_distance_matrix(pairwise_distance_ts_pred, atom_count)
    initial = classical_mds(distance_matrix, dimensions=3)
    reference = 0.5 * (reactant_pos + product_pos)
    initial = align_to_reference(initial, reference).detach()

    mask = ~torch.eye(atom_count, dtype=torch.bool, device=reactant_pos.device)
    src, dst = torch.where(mask)
    target_distances = distance_matrix[src, dst].detach()

    coordinates = initial.clone().requires_grad_(True)
    optimizer = LBFGS([coordinates], max_iter=max_iter, lr=lr)

    def closure():
        optimizer.zero_grad()
        loss = weighted_distance_loss(
            coordinates, target_distances, src, dst, epsilon=epsilon
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    loss = weighted_distance_loss(
        coordinates, target_distances, src, dst, epsilon=epsilon
    )
    return coordinates.detach(), loss.detach()


def pairwise_dist_to_coord_linear_interp(
    reactant_pos,
    product_pos,
    pairwise_distance_ts_pred,
    max_iter=100,
    lr=0.1,
    epsilon=1e-6,
):
    if reactant_pos.shape != product_pos.shape or reactant_pos.ndim != 2:
        raise ValueError("reactant_pos and product_pos must have matching [N, D] shapes")
    if reactant_pos.shape[1] != 3:
        raise ValueError("linear-interpolation reconstruction requires 3D inputs")
    if max_iter <= 0:
        raise ValueError("max_iter must be greater than zero")
    if lr <= 0:
        raise ValueError("lr must be greater than zero")

    atom_count = reactant_pos.shape[0]
    distance_matrix = complete_distance_matrix(pairwise_distance_ts_pred, atom_count)
    initial = 0.5 * (reactant_pos + product_pos)

    mask = ~torch.eye(atom_count, dtype=torch.bool, device=reactant_pos.device)
    src, dst = torch.where(mask)
    target_distances = distance_matrix[src, dst].detach()

    coordinates = initial.clone().detach().requires_grad_(True)
    optimizer = LBFGS([coordinates], max_iter=max_iter, lr=lr)

    def closure():
        optimizer.zero_grad()
        loss = weighted_distance_loss(
            coordinates, target_distances, src, dst, epsilon=epsilon
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    loss = weighted_distance_loss(
        coordinates, target_distances, src, dst, epsilon=epsilon
    )
    return coordinates.detach(), loss.detach()
