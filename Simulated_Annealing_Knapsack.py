import numpy as np
import math
import random
import matplotlib.pyplot as plt

random.seed(0)
np.random.seed(0)

items = np.load("knapsack_dataset_1.npy")
n = items.shape[0]
max_weight = 3500

def initial_solution(items, max_weight, n):
    weight = 0
    sol = []
    for i in range(n):
        if (weight + items[i,1] > max_weight):
            break
        else:
            sol.append(i)
            weight += items[i,1]
    return sol, weight

        
def step(cur_sol, n, max_weight, items, cur_weight, cur_val):
    r = random.randint(0, n-1)
    while(r in cur_sol):
        r = random.randint(0, n-1)
    new_sol = cur_sol.copy()
    new_sol.append(r)
    new_weight = cur_weight + items[r,1]
    new_val = cur_val + items[r,0]
    dif = new_weight - max_weight
    l = len(cur_sol)
    while (new_weight - max_weight > 0):
        r = random.randint(0, l)
        i = new_sol[r]
        new_sol.pop(r)
        new_weight -= items[i,1]
        new_val -= items[i,0]
        l -= 1
    if (new_val >= cur_val):
        return new_sol, new_val, new_weight   #if new solution is better, automatically accept
    else:
        rand = random.random()
        prob = math.exp((new_val - cur_val) / temp)
        if (rand <= prob):
            return new_sol, new_val, new_weight  #accept worse solution with given probability
        else:
            return cur_sol, cur_val, cur_weight


#different_decays = list(range(80,99,1)) #exponential annealing schedule
different_decays = [0.87]
different_temps = [1, 10, 100, 1000, 5000]
different_temps = [100]
results = []
temperatures = []
best_solutions = []
init_sol, init_weight = initial_solution(items, max_weight, n)
for temp_decay in different_decays:
    temp_decay *= 0.01
    for temp in different_temps:
        res = []
        temps = []
        sol = init_sol
        cur_weight = init_weight
        cur_val = 0
        max_sol = 0
        for i in sol:
            cur_val += items[i,0]
        num_iterations = 1000
        for i in range(num_iterations):
            temps.append(temp)
            sol, cur_val, cur_weight = step(sol, n, max_weight, items, cur_weight, cur_val)
            temp *= temp_decay
            if (max_sol < cur_val):
                max_sol = cur_val
            res.append(max_sol)
        print(max_sol)
        results.append(res)
        best_solutions.append(max_sol)
        temperatures.append(temps)

x_arr = np.arange(1,num_iterations+1,1, dtype=int)
plt.plot(x_arr,results[0])


"""
bs = np.zeros((19,1))
bs.fill(155999)
plt.plot(different_decays, best_solutions[0:19], label="T=1")
plt.plot(different_decays, best_solutions[19:38], label="T=10")
plt.plot(different_decays, best_solutions[38:57], label="T=100")
plt.plot(different_decays, best_solutions[57:76], label="T=1000")
plt.plot(different_decays, best_solutions[76: 95], label="T=5000")
#plt.plot(different_decays,bs, 'g-', label="optimal solution")
plt.legend()
"""
plt.xlabel("Iterations")
plt.ylabel("Best Solution")
plt.show()
plt.savefig("simulated_annealing_plot_large_0.90.jpg")

