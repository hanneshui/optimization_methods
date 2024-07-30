import numpy as np


def dp_solution_knapsack(items, n, l):
    # create a table to store the maximum value that can be obtained with the first i items and a weight of j
    table = np.zeros((n + 1, l + 1))
    for i in range(1, n + 1):
        for j in range(1, l + 1):
            # if the weight of the current item is less than or equal to the current weight limit
            if items[i - 1][1] <= j:
                # take the maximum of including the current item or excluding the current item
                table[i][j] = max(
                    table[i - 1][j],
                    table[i - 1][j - items[i - 1][1]] + items[i - 1][0],
                )
            else:
                # if the weight of the current item is greater than the current weight limit, exclude the current item
                table[i][j] = table[i - 1][j]
    return table[n][l]


datasets = [
    {"filename": "knapsack_dataset_1.npy", "limit": 3500},
    {"filename": "knapsack_dataset_2.npy", "limit": 35000},
    {"filename": "knapsack_dataset_3.npy", "limit": 75000},
]

for dataset in datasets:
    items = np.load(dataset["filename"])
    n = items.shape[0]
    l = dataset["limit"]

    solution = dp_solution_knapsack(items, n, l)
    print(f"Solution for {dataset['filename']}: {solution}")
