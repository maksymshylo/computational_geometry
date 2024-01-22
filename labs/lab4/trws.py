import numpy as np
from numba import njit
from numpy import ndarray


@njit(fastmath=True)
def init_q(images: ndarray) -> ndarray:
    """
    Initialize unary penalties.
    0 - if (i,j) can take from k-image, else -inf

    Args:
        images: Input images
    Returns
        Unary penalties
    """
    n_images = len(images)
    Q = np.full((images[0].shape[0], images[0].shape[1], n_images), -np.inf)
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            for k in range(Q.shape[2]):
                if (images[k][i, j] != 0).any():
                    Q[i, j, k] = 0
            if (images[:, i, j, :] == 0).all():
                Q[i, j, :] = 0

    return Q


@njit(fastmath=True)
def custom_norm(vec: ndarray) -> ndarray:
    """L2 - norm for 3D vector."""
    return np.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)


@njit(fastmath=True)
def init_g(images: ndarray) -> ndarray:
    """
    Initialize binary penalties.

    Args:
        images: Input images
    Returns
        Binary penalties
    """
    height, width, _ = images[0].shape
    n_images = len(images)
    g = np.zeros((height, width, 4, n_images, n_images))

    # for each pixel in image
    for i in range(height):
        for j in range(width):
            nbs, _, nbs_indices = get_neighbours(height, width, i, j)
            # for each neighbour
            for n, [n_i, n_j] in enumerate(nbs):
                for k in range(n_images):
                    for k1 in range(n_images):
                        if k != k1:
                            g[i, j, nbs_indices[n], k, k1] = -custom_norm(
                                images[k, i, j] - images[k1, i, j]
                            ) - custom_norm(images[k, n_i, n_j] - images[k1, n_i, n_j])

    return g


@njit
def get_neighbours(height: int, width: int, i: int, j: int) -> tuple[list, list, list]:
    """
    Calculate neighbours in 4-neighbours system (and inverse indices).
            (i-1,j)
               |
    (i,j-1)--(i,j)--(i,j+1)
               |
            (i+1,j)

    Args:
        height: Height of input image
        width: Width of input image
        i: row
        j: column
    Returns
        Neighbours coordinates, neighbours indices, inverse neighbours indices
    """
    nbs = []
    # neighbour indices
    nbs_indices = []
    # inverse neighbour indices
    inv_nbs_indices = []
    # Left
    if 0 <= j - 1 < width - 1 and 0 <= i <= height - 1:
        nbs.append([i, j - 1])
        inv_nbs_indices.append(1)
        nbs_indices.append(0)
    # Right
    if 0 < j + 1 <= width - 1 and 0 <= i <= height - 1:
        nbs.append([i, j + 1])
        inv_nbs_indices.append(0)
        nbs_indices.append(1)
    # Upper
    if 0 <= i - 1 < height - 1 and 0 <= j <= width - 1:
        nbs.append([i - 1, j])
        inv_nbs_indices.append(3)
        nbs_indices.append(2)
    # Down
    if 0 < i + 1 <= height - 1 and 0 <= j <= width - 1:
        nbs.append([i + 1, j])
        inv_nbs_indices.append(2)
        nbs_indices.append(3)

    neighbours_data = (nbs, inv_nbs_indices, nbs_indices)

    return neighbours_data


@njit(fastmath=True, cache=True)
def forward_pass(
    height: int,
    width: int,
    n_labels: int,
    Q: ndarray,
    g: ndarray,
    P: ndarray,
    fi: ndarray,
) -> tuple[ndarray, ndarray]:
    """
    Update fi according to P for 'Left' and 'Up' directions

    Args:
        height: Height of input image
        width: Width of input image
        n_labels: Number of labels
        Q: Unary penalties
        g: Binary penalties
        P: The best path weight for each direction (Left,Right,Up,Down)
        fi: Potentials

    Returns
        Updated P and fi
    """
    # for each pixel of input channel
    for i in range(1, height):
        for j in range(1, width):
            # for each label in pixel
            for k in range(n_labels):
                # P[i,j,0,k] - Left direction
                # P[i,j,2,k] - Up direction
                # calculate the best path weight according to formula
                P[i, j, 0, k] = max(
                    P[i, j - 1, 0, :]
                    + (1 / 2) * Q[i, j - 1, :]
                    - fi[i, j - 1, :]
                    + g[i, j - 1, 1, :, k]
                )
                P[i, j, 2, k] = max(
                    P[i - 1, j, 2, :]
                    + (1 / 2) * Q[i - 1, j, :]
                    + fi[i - 1, j, :]
                    + g[i - 1, j, 3, :, k]
                )
                # update potentials
                fi[i, j, k] = (
                    P[i, j, 0, k] + P[i, j, 1, k] - P[i, j, 2, k] - P[i, j, 3, k]
                ) / 2

    return (P, fi)


@njit(fastmath=True, cache=True)
def backward_pass(
    height: int,
    width: int,
    n_labels: int,
    Q: ndarray,
    g: ndarray,
    P: ndarray,
    fi: ndarray,
) -> tuple[ndarray, ndarray]:
    """
    Update fi according to P for 'Right' and 'Down' directions

    Args:
        height: Height of input image
        width: Width of input image
        n_labels: Number of labels
        Q: Unary penalties
        g: Binary penalties
        P: The best path weight for each direction (Left,Right,Up,Down)
        fi: Potentials

    Returns
        Updated P and fi
    """
    # for each pixel of input channel
    # go from bottom-right to top-left pixel
    for i in np.arange(height - 2, -1, -1):
        for j in np.arange(width - 2, -1, -1):
            # for each label in pixel
            for k in range(n_labels):
                # P[i,j,1,k] - Right direction
                # P[i,j,3,k] - Down direction
                # calculate the best path weight according to formula
                P[i, j, 3, k] = max(
                    P[i + 1, j, 3, :]
                    + (1 / 2) * Q[i + 1, j, :]
                    + fi[i + 1, j, :]
                    + g[i + 1, j, 2, k, :]
                )
                P[i, j, 1, k] = max(
                    P[i, j + 1, 1, :]
                    + (1 / 2) * Q[i, j + 1, :]
                    - fi[i, j + 1, :]
                    + g[i, j + 1, 0, k, :]
                )
                # update potentials
                fi[i, j, k] = (
                    P[i, j, 0, k] + P[i, j, 1, k] - P[i, j, 2, k] - P[i, j, 3, k]
                ) / 2

    return (P, fi)


def trws(
    height: int,
    width: int,
    n_labels: int,
    Q: ndarray,
    g: ndarray,
    P: ndarray,
    n_iter: int,
) -> ndarray:
    """
    Run TRW-S algorithm (forward and backward pass) n_iter times.
    Update fi according to P.

    Args:
        height: Height of input image
        width: Width of input image
        n_labels: Number of labels
        Q: Unary penalties
        g: Binary penalties
        P: The best path weight for each direction (Left,Right,Up,Down)
        n_iter: Number of iterations
    Returns
        Optimal labelling (with color mapping)
    """
    # initialise array of potentials with zeros
    fi = np.zeros((height, width, n_labels))
    # initialize Right and Down directions
    P, _ = backward_pass(height, width, n_labels, Q, g, P, fi.copy())
    for _iter in range(n_iter):
        P, fi = forward_pass(height, width, n_labels, Q, g, P, fi)
        P, fi = backward_pass(height, width, n_labels, Q, g, P, fi)

    # restore labelling from optimal energy after n_iter of TRW-S
    labelling = np.argmax(P[:, :, 0, :] + P[:, :, 1, :] - fi + Q / 2, axis=2)

    return labelling


def create_panorama(aligned_images: list, n_iter: int) -> ndarray:
    """
    Create a panorama with TRW-S algorithm as a stitching method.

    Args:
        images: Aligned images
        Q: Unary penalties
        g: Binary penalties
        n_iter: Number of iterations
    Returns
        Stitched image
    """

    @njit(fastmath=True)
    def map_labels(images: ndarray, labelling: ndarray) -> ndarray:
        """
        Map proper pixels from labelling to panorama image.

        Args:
            images: Aligned images.
            labelling: Optimal labelling (with color mapping).
                       Result from TRWS algorithm.

        Returns
            Stitched image
        """
        panorama = np.zeros_like(images[0])
        for i in range(panorama.shape[0]):
            for j in range(panorama.shape[1]):
                # get pixel from aligned image based on labels
                panorama[i, j, :] = images[labelling[i, j], i, j, :]

        return panorama.astype(np.int32)

    images = np.array(aligned_images, dtype=np.float32)
    # initialize unary and binary penalties
    Q = init_q(images)
    g = init_g(images)
    # initialize the best path weight for each direction (Left,Right,Up,Down)
    height, width, n_labels = Q.shape
    P = np.zeros((height, width, 4, n_labels))
    # run trws algorithm
    labelling = trws(height, width, n_labels, Q, g, P, n_iter)
    # map pixels to panorama
    panorama = map_labels(images, labelling)

    return panorama
