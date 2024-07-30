import numpy as np
import scipy.stats as stats

np.random.seed(0)
stats.truncnorm.random_state = np.random.RandomState(0)


def generate_dataset(num_items, weight_range, limit, filename):
    weights = np.random.randint(
        low=weight_range[0], high=weight_range[1] + 1, size=num_items
    )

    mu, sigma = 1, 1
    lower, upper = 0.5, 5
    truncated_normal = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma
    )
    x_values = truncated_normal.rvs(num_items)

    values = np.round(weights * x_values).astype(int)

    items = np.column_stack((values, weights))

    np.save(filename, items)
    print(f"Dataset saved to {filename} with shape {items.shape}")


datasets = [
    {
        "num_items": 100,
        "weight_range": (1, 100),
        "limit": 3500,
        "filename": "knapsack_dataset_1.npy",
    },
    {
        "num_items": 1000,
        "weight_range": (1, 100),
        "limit": 35000,
        "filename": "knapsack_dataset_2.npy",
    },
    {
        "num_items": 3000,
        "weight_range": (1, 100),
        "limit": 75000,
        "filename": "knapsack_dataset_3.npy",
    },
]

for dataset in datasets:
    generate_dataset(
        dataset["num_items"],
        dataset["weight_range"],
        dataset["limit"],
        dataset["filename"],
    )
