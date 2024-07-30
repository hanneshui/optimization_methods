import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

datasets = [
    {
        "filename": "knapsack_dataset_1.npy",
        "limit": 3500,
        "optimal_value": 6228,
        "k": 250,
    },
    {
        "filename": "knapsack_dataset_2.npy",
        "limit": 35000,
        "optimal_value": 64409,
        "k": 600,
    },
    {
        "filename": "knapsack_dataset_3.npy",
        "limit": 75000,
        "optimal_value": 155999,
        "k": 1000,
    },
]


# fitness function to calculate the value of the knapsack for a given individual
def fitness_function(items, individual, limit):
    weight = 0
    value = 0
    for i in range(n):
        if individual[i] == 1:
            weight += items[i][1]
            value += items[i][0]
    if weight > limit:

        return 0
    return value


# crossover function to combine two parents to create two children using one point crossover
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, n)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2


# mutation function to flip a random bit in the individual
def mutate(individual):
    # flip each bit with a probability of 1/n
    for i in range(n):
        if np.random.rand() < 1 / n:
            individual[i] = 1 - individual[i]
    return individual


# genetic algorithm function to find the best solution
def genetic_algorithm(items, n, limit, k):
    avg_fitness = []
    best_fitness = []
    diversity = []
    fitness_distribution = []
    # creating a random population of size 10
    population_size = 50
    # population = np.random.randint(2, size=(population_size, n))
    # population with all zeros
    population = np.zeros((population_size, n))
    # randomly mutate each bit with a probability of 1/n
    for i in range(population_size):
        population[i] = mutate(population[i])

    # avg_fitness.append(
    #   np.mean([fitness_function(individual) for individual in population])
    # )

    # running the genetic algorithm for 100 generations
    for i in range(k):
        # selection based on a tournament of 4 individuals (always select the best 2)
        # first calculate the fitness of each individual and sort the population based on fitness
        fitness = np.array(
            [fitness_function(items, individual, limit) for individual in population]
        )
        population = population[np.argsort(fitness)[::-1]]
        population = population[: population_size // 2]
        # reproduce the top 10% best individuals to the next generation
        new_population = [
            individual
            for individual, _ in zip(population, range(population_size // 10))
        ]

        # crossover and mutation to create new individuals with random selection of parents
        for j in range((population_size // 2) - 1):
            parent1 = population[np.random.randint(population_size // 2)]
            parent2 = population[np.random.randint(population_size // 2)]
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))
        population = np.array(new_population)
        best_fitness.append(fitness_function(items, population[0], limit))
        unique_individuals = len(np.unique(population, axis=0))
        diversity.append(unique_individuals / population.shape[0])
        fitness_distribution.append(fitness)

        avg_fitness.append(
            np.mean(
                [
                    fitness_function(items, individual, limit)
                    for individual in population
                ]
            )
        )
        if i % 100 == 0:
            print("Generation", i, "Average Fitness:", avg_fitness[-1])
        # stop if 50 last average fitness values are the same
        if len(avg_fitness) > 50 and len(set(avg_fitness[-50:])) == 1:
            print("Converged after", i + 1, "generations")
            break
    # calculate the fittest individual in the final population
    fitness = np.array(
        [fitness_function(items, individual, limit) for individual in population]
    )
    population = population[np.argsort(fitness)[::-1]]

    return (
        population[0],
        avg_fitness,
        best_fitness,
        diversity,
        fitness_distribution,
        population,
    )


for dataset in datasets:
    items = np.load(dataset["filename"])
    n = items.shape[0]
    l = dataset["limit"]
    k = dataset["k"]

    solution, avg_fitness, best_fitness, diversity, fitness_distribution, population = (
        genetic_algorithm(items, n, l, k)
    )

    # ploting
    print(f"Best Fitness: {fitness_function(items, solution, l)}")

    # Plotting results
    generations = range(len(avg_fitness))
    optimal_value = dataset["optimal_value"]
    plt.figure(figsize=(12, 6))
    plt.plot(generations, avg_fitness, label="Average Fitness")
    plt.plot(generations, best_fitness, label="Best Fitness")
    plt.plot(
        generations, [optimal_value] * len(generations), label="Optimal Value from DP"
    )
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness over Generations")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(generations, diversity, label="Population Diversity")
    plt.xlabel("Generation")
    plt.ylabel("Procentage of Unique Individuals")
    plt.title("Diversity over Generations")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot fitness distribution for the first and last generations
    plt.figure(figsize=(12, 6))
    plt.hist(fitness_distribution[0], bins=50, alpha=0.5, label="Generation 0")
    plt.hist(fitness_distribution[20], bins=50, alpha=0.5, label="Generation 20")
    plt.hist(fitness_distribution[50], bins=50, alpha=0.5, label="Generation 50")
    # plt.hist(fitness_distribution[-1], bins=50, alpha=0.5, label="Last Generation")
    plt.xlabel("Fitness")
    plt.ylabel("Frequency")
    plt.title("Fitness Distribution selected Generations")
    plt.legend()
    plt.grid(True)
    plt.show()
