import numpy as np
import pandas as pd


def validate_covariance(covariance):
    """Validate and return covariance matrix as NumPy array."""

    covariance = np.asarray(
        covariance,
        dtype=float
    )

    if covariance.ndim != 2:
        raise ValueError(
            "covariance must be a 2-dimensional matrix"
        )

    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            "covariance matrix must be square"
        )

    if covariance.shape[0] == 0:
        raise ValueError(
            "covariance matrix cannot be empty"
        )

    if not np.all(np.isfinite(covariance)):
        raise ValueError(
            "covariance matrix must contain finite values"
        )

    if not np.allclose(
        covariance,
        covariance.T,
        atol=1e-10
    ):
        raise ValueError(
            "covariance matrix must be symmetric"
        )

    return covariance


def covariance_to_correlation(covariance):
    """
    Convert covariance matrix to correlation matrix.
    """

    covariance = validate_covariance(
        covariance
    )

    volatility = np.sqrt(
        np.maximum(
            np.diag(covariance),
            1e-16
        )
    )

    correlation = (
        covariance
        / np.outer(
            volatility,
            volatility
        )
    )

    correlation = np.clip(
        correlation,
        -1.0,
        1.0
    )

    np.fill_diagonal(
        correlation,
        1.0
    )

    return correlation


def correlation_distance(correlation):
    """
    Convert correlation into distance.

    Formula:

        d(i,j) = sqrt((1 - rho(i,j)) / 2)
    """

    correlation = np.asarray(
        correlation,
        dtype=float
    )

    if correlation.ndim != 2:
        raise ValueError(
            "correlation must be 2-dimensional"
        )

    if (
        correlation.shape[0]
        != correlation.shape[1]
    ):
        raise ValueError(
            "correlation matrix must be square"
        )

    distance = np.sqrt(
        np.maximum(
            (1.0 - correlation) / 2.0,
            0.0
        )
    )

    np.fill_diagonal(
        distance,
        0.0
    )

    return distance


def hierarchical_clustering_order(correlation):
    """
    Create an asset ordering using hierarchical clustering.

    This is implemented from scratch using an
    agglomerative clustering procedure.

    At every step, the two closest clusters are merged.
    """

    distance = correlation_distance(
        correlation
    )

    n_assets = distance.shape[0]

    if n_assets == 1:
        return [0]

    clusters = [
        [i]
        for i in range(n_assets)
    ]

    # ---------------------------------------------------------
    # Cluster distance:
    #
    # Average pairwise distance between assets
    # in the two clusters.
    # ---------------------------------------------------------

    def cluster_distance(
        cluster_a,
        cluster_b
    ):
        values = []

        for i in cluster_a:
            for j in cluster_b:
                values.append(
                    distance[i, j]
                )

        return float(
            np.mean(values)
        )

    # ---------------------------------------------------------
    # Agglomerative clustering
    # ---------------------------------------------------------

    while len(clusters) > 1:

        best_distance = np.inf
        best_pair = None

        for i in range(
            len(clusters)
        ):

            for j in range(
                i + 1,
                len(clusters)
            ):

                current_distance = (
                    cluster_distance(
                        clusters[i],
                        clusters[j]
                    )
                )

                if current_distance < best_distance:

                    best_distance = (
                        current_distance
                    )

                    best_pair = (
                        i,
                        j
                    )

        i, j = best_pair

        merged_cluster = (
            clusters[i]
            + clusters[j]
        )

        # Remove larger index first.
        clusters.pop(j)
        clusters.pop(i)

        clusters.append(
            merged_cluster
        )

    return clusters[0]


def cluster_variance(
    cluster,
    covariance
):
    """
    Calculate inverse-variance portfolio variance
    inside a cluster.
    """

    covariance = validate_covariance(
        covariance
    )

    cluster = list(cluster)

    if len(cluster) == 0:
        raise ValueError(
            "cluster cannot be empty"
        )

    cluster_covariance = covariance[
        np.ix_(
            cluster,
            cluster
        )
    ]

    variances = np.diag(
        cluster_covariance
    )

    inverse_variance = (
        1.0
        / np.maximum(
            variances,
            1e-16
        )
    )

    weights = (
        inverse_variance
        / inverse_variance.sum()
    )

    variance = (
        weights
        @ cluster_covariance
        @ weights
    )

    return float(
        max(variance, 0.0)
    )


def quasi_diagonal_order(correlation):
    """
    Return a quasi-diagonalized asset ordering.

    Assets that are more closely related are placed
    near one another.
    """

    order = hierarchical_clustering_order(
        correlation
    )

    return order


def hrp_weights(covariance):
    """
    Calculate Hierarchical Risk Parity weights.

    HRP process:

        1. Convert covariance to correlation.
        2. Build hierarchical clusters.
        3. Order assets.
        4. Allocate capital recursively between clusters.
        5. Use inverse-variance allocation within clusters.

    Returns
    -------
    np.ndarray
        HRP portfolio weights.
    """

    covariance = validate_covariance(
        covariance
    )

    n_assets = covariance.shape[0]

    if n_assets == 1:
        return np.array([1.0])

    correlation = covariance_to_correlation(
        covariance
    )

    order = quasi_diagonal_order(
        correlation
    )

    weights = np.ones(
        n_assets,
        dtype=float
    )

    # ---------------------------------------------------------
    # Recursive bisection
    # ---------------------------------------------------------

    clusters = [order]

    while clusters:

        next_clusters = []

        for cluster in clusters:

            if len(cluster) <= 1:
                continue

            split = len(cluster) // 2

            left_cluster = (
                cluster[:split]
            )

            right_cluster = (
                cluster[split:]
            )

            left_variance = cluster_variance(
                left_cluster,
                covariance
            )

            right_variance = cluster_variance(
                right_cluster,
                covariance
            )

            total_variance = (
                left_variance
                + right_variance
            )

            if total_variance <= 1e-16:

                left_allocation = 0.5
                right_allocation = 0.5

            else:

                # Allocate more capital to
                # the lower-risk cluster.
                left_allocation = (
                    right_variance
                    / total_variance
                )

                right_allocation = (
                    left_variance
                    / total_variance
                )

            for asset in left_cluster:
                weights[asset] *= (
                    left_allocation
                )

            for asset in right_cluster:
                weights[asset] *= (
                    right_allocation
                )

            next_clusters.extend(
                [
                    left_cluster,
                    right_cluster
                ]
            )

        clusters = next_clusters

    # ---------------------------------------------------------
    # Final normalization
    # ---------------------------------------------------------

    weights = np.maximum(
        weights,
        0.0
    )

    weights = (
        weights / weights.sum()
    )

    return weights


def hrp_series(
    covariance,
    asset_names=None
):
    """
    Calculate HRP weights as a pandas Series.
    """

    covariance_array = validate_covariance(
        covariance
    )

    n_assets = covariance_array.shape[0]

    if asset_names is None:

        if isinstance(
            covariance,
            pd.DataFrame
        ):
            asset_names = (
                covariance.columns.tolist()
            )

        else:
            asset_names = [
                f"Asset_{i}"
                for i in range(n_assets)
            ]

    if len(asset_names) != n_assets:
        raise ValueError(
            "asset_names must match "
            "number of assets"
        )

    weights = hrp_weights(
        covariance
    )

    return pd.Series(
        weights,
        index=asset_names,
        name="Weight"
    )


def hrp_dataframe(
    covariance,
    asset_names=None
):
    """
    Return HRP portfolio as a DataFrame.
    """

    weights = hrp_series(
        covariance,
        asset_names
    )

    return weights.to_frame()


def validate_hrp_weights(
    weights,
    tolerance=1e-10
):
    """
    Validate HRP portfolio weights.
    """

    weights = np.asarray(
        weights,
        dtype=float
    )

    if weights.ndim != 1:
        raise ValueError(
            "weights must be 1-dimensional"
        )

    if len(weights) == 0:
        raise ValueError(
            "weights cannot be empty"
        )

    if not np.all(np.isfinite(weights)):
        raise ValueError(
            "weights must contain finite values"
        )

    if np.any(weights < -tolerance):
        return False

    return bool(
        np.isclose(
            weights.sum(),
            1.0,
            atol=tolerance
        )
    )