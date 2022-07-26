import torch


def sampling_layer(points, n_sampled):
    """
    :param points: B X N X F - 'N' points in an 'F' dimensional space, the point cloud
    :param n_sampled: number of sampled points wanted
    :return: function returns sampled points from the point cloud using farthest point sampling
    """
    # ----------------------------------------------------------------------------------------------------------
    # Local variables
    # ----------------------------------------------------------------------------------------------------------
    device                         = points.device
    batch_size, n_points, dim_size = points.shape
    sampled_points                 = torch.zeros(batch_size, n_sampled, dtype=torch.long).to(device)
    distance                       = torch.ones(batch_size, n_points).to(device) * float('inf')  # init distance to infinity
    batch_idx                      = torch.arange(batch_size, dtype=torch.long).to(device)
    # ----------------------------------------------------------------------------------------------------------
    # Choosing first point at random
    # ----------------------------------------------------------------------------------------------------------
    farthest_point       = torch.randint(0, n_points, (batch_size,), dtype=torch.long).to(device)
    sampled_points[:, 0] = farthest_point
    # ----------------------------------------------------------------------------------------------------------
    # Iterating and choosing farthest
    # ----------------------------------------------------------------------------------------------------------
    for ii in range(1, n_sampled):
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Computing distances from last selected point
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        centroid = points[batch_idx, farthest_point, :].view(batch_size, 1, dim_size)
        dist     = torch.sum((points - centroid)**2, -1)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Updating new minimal distance
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        dist_mask = dist < distance
        distance[dist_mask] = dist[dist_mask]
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Choosing new farthest point
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        farthest_point = torch.max(distance, -1)[1]
        sampled_points[:, ii] = farthest_point
    return sampled_points


def grouping_layer(radius, k, points, centroids, return_mask=False):
    """
    :param radius: Radius of the convolution
    :param k: Maximal samples per region (for the convolution)
    :param points: B X N X F - 'N' points in an 'F' dimensional space, the point cloud
    :param centroids: B X N' X F
    :param return_mask: if True, returns the mask
    :return: B X N' X K tensor containing the indices of the indices for each centroid
    """
    # ----------------------------------------------------------------------------------------------------------
    # Local variables
    # ----------------------------------------------------------------------------------------------------------
    device = points.device
    batch_size, n_points, dim_size = points.shape
    _, ntag_points, _              = centroids.shape
    square_dist                    = _square_distance(points, centroids)
    # ----------------------------------------------------------------------------------------------------------
    # group_idx holds the allocation of each of the coordinates to one of the centroids, up to K neighbors
    # ----------------------------------------------------------------------------------------------------------
    group_idx = torch.arange(n_points, dtype=torch.long).to(device).view(1, 1, n_points).repeat([batch_size, ntag_points, 1])
    group_idx[square_dist > radius ** 2] = n_points  # allocating all the points outside the radius from a centroid to the last group
    group_idx = group_idx.sort(dim=-1)[0]
    if k is None:
        k = (group_idx != n_points).sum(dim=-1).max().item()
    group_idx = group_idx[:, :, :k]
    # ----------------------------------------------------------------------------------------------------------
    # masking the out of bounds if needed
    # ----------------------------------------------------------------------------------------------------------
    mask = group_idx == n_points
    group_first = group_idx[:, :, 0].view(batch_size, ntag_points, 1).repeat([1, 1, k])
    group_idx[mask] = group_first[mask]

    if return_mask:
        return group_idx, mask
    return group_idx


def sample_group(ntag_points, radius, k, points, data):
    """
    :param ntag_points:
    :param radius:
    :param k: maximal points per group. If None, takes into consideration all neighbors in the radius
    :param points:  B X N X F matrix of coordinates, F = 2,3
    :param data: B X N X D matrix of the data for each point
    :return:
        centroids: B X N' X F coordinates of the centroids
        grouped_data: B X N' X K X (F + D) tensor containing the coordinates and data of each group
    """
    # ----------------------------------------------------------------------------------------------------------
    # Local variables
    # ----------------------------------------------------------------------------------------------------------
    batch_size, n_points, dim_size = points.shape
    # ----------------------------------------------------------------------------------------------------------
    # Sampling
    # ----------------------------------------------------------------------------------------------------------
    centroids_idx = sampling_layer(points, ntag_points)  # B X N' dimension
    centroids     = _idx2data(points, centroids_idx)     # B X N' X F dimension
    # ----------------------------------------------------------------------------------------------------------
    # Grouping
    # ----------------------------------------------------------------------------------------------------------
    group_idx, mask     = grouping_layer(radius, k, points, centroids, return_mask=True)
    grouped_points      = _idx2data(points, group_idx)
    grouped_points_norm = grouped_points - centroids.view(batch_size, ntag_points, 1, dim_size)
    grouped_data        = _idx2data(data, group_idx)
    grouped_total       = torch.cat([grouped_points_norm, grouped_data], dim=-1)  # B X N' X K X (F + D)
    grouped_total[mask] = 0.0

    return centroids, grouped_total


def knn(x, k):
    """
    :param x: B X N X F matrix of coordinates
    :param k: number of nearest neighbors wanted
    :return: performing KNN for each of the coordinates
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


# ==================================================================================================================
# Auxiliary
# ==================================================================================================================
def _square_distance(points1, points2):
    """
    :param points1: B X N X F  --> F = 2
    :param points2: B X N' X F
    :return: B X N X N' per-point square distance
    """
    batch_size  = points1.shape[0]
    points1_per = points1.permute(0, 2, 1)
    distance    = torch.zeros((batch_size, points2.shape[1], points1.shape[1]))
    for ii in range(points1.shape[2]):
        distance += (points2[:, :, ii][:, :, None] - points1_per[:, ii, :]) ** 2
    return distance


def _idx2data(points, idx_mat):
    """
    :param points: B X N X F
    :param idx_mat: B X N' index matrix
    :return: B X N' X F matrix with the data prom points, as depicted in idx
    """
    device     = points.device
    batch_size = points.shape[0]

    view_shape     = list(idx_mat.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)

    repeat_shape    = list(idx_mat.shape)
    repeat_shape[0] = 1

    indices = torch.arange(batch_size, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    return points[indices, idx_mat, :]

