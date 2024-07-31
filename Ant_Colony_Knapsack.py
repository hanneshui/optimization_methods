import numpy as np
import matplotlib.pyplot as plt

import random
random.seed(0)
np.random.seed(0)

items = np.load("knapsack_dataset_1.npy")
n = items.shape[0]
max_weight = 3500
#pheromone update
def update(paths, best_sol, ants, num_ants, n):
    pheromone_upd = np.zeros((n+1,n+1))
    for a in range(num_ants):
        path = paths[a]
        val = ants[a,0]
        for i in range(len(path) - 1):
            n1 = path[i]
            n2 = path[i+1]
            pheromone_upd[n1,n2] += val / (best_sol)
    return pheromone_upd

def calc_attractiveness(items, pheromone_arr, alpha, beta, n):
    A = np.zeros(pheromone_arr.shape)
    for i in range(n+1):
        for j in range(n):
            A[i,j] = (pheromone_arr[i,j] ** alpha) * ((items[j,0] / items[j,1]) ** beta)

    return A

def decision(A, paths, ant_id, ant, items, max_weight):
    probs = [0]
    options = []
    sum_ = 0
    path = paths[ant_id]
    cur_location = path[-1]
    cur_weight = ant[1]
    S = set(range(1,n+1)).difference(path)
    for i in S:
        if (cur_weight + items[i,1] <= max_weight and i != cur_location and A[cur_location,i] > 0):
            options.append(i)
            sum_ += A[cur_location, i]
            probs.append(sum_)
    if (sum_ == 0):
        return False
    num = random.random() * sum_
    sol = 0
    for j in range(len(probs)):
        prob = probs[j]
        if (num < prob):
            sol = options[j-1]
            paths[ant_id].append(sol)
            ant[0] += items[sol,0]
            ant[1] += items[sol,1]
            break
    return True

start_pheromone = 1
sigma = 0.5 #pheromone factor
alpha= 0.5
beta = 2
results = []
ant_sizes = [10,20,30,40,50]
best_solutions = []
for num_ants in ant_sizes:
    num_iterations = 100
    #do first 
    pheromone_arr = np.zeros((n+1,n+1))
    pheromone_arr.fill(start_pheromone)
    best_sol = 0
    res = []
    for k in range(num_iterations):
        ants = np.zeros((num_ants,2))
        paths = []
        active = np.full((num_ants), True)
        A = calc_attractiveness(items, pheromone_arr, alpha, beta, n)
        for i in range(num_ants):
            paths.append([n])
        while (True in active):
            for a in range(num_ants):
                flag = decision(A, paths, a, ants[a], items, max_weight)
                active[a] = flag
        max = np.max(ants[:,0], axis = 0).item()
        if (max > best_sol):
            best_sol = max
        res.append(best_sol)
        pheromone_upd = update(paths, best_sol, ants, num_ants, n)
        pheromone_arr = pheromone_arr * sigma + pheromone_upd
        #print(max)
        #print(k)
    results.append(res)
    best_solutions.append(best_sol)
    print("done")
print(best_sol)

x_arr = np.arange(1,num_iterations+1,1, dtype=int)
bs = np.zeros((5,1))
bs.fill(6228)
"""
plt.plot(x_arr,bs, 'g-', label="optimal solution")
plt.plot(x_arr,results[0], label="number of ants = 10")
plt.plot(x_arr,results[1], label="number of ants = 20")
plt.plot(x_arr,results[2], label="number of ants = 30")
plt.plot(x_arr,results[3], label="number of ants = 40")
plt.plot(x_arr,results[4], label="number of ants = 50")
plt.legend()
"""
#plt.plot(ant_sizes,bs, 'g-', label="optimal solution")
plt.plot(ant_sizes, best_solutions, 'bo', linestyle='solid', label="AC solution")
plt.xlabel("Number of Ants")
plt.ylabel("Best Solution")
#plt.legend()
plt.show()
plt.savefig("ant_colony_medium_alphas_betas.jpg")
#print(pheromone_arr)