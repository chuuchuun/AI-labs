from itertools import compress
import random
import time
import matplotlib.pyplot as plt
import numpy as np

from data import *

def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]

def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))

def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness


items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
generations = 200
n_selection = 20
n_elite = 1
p_Mut = 0.01
start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)


def roulette_wheel_selection(population, n_selection):
    f_j = sum([fitness(items, knapsack_max_capacity,individual) for individual in population])
    probability = [fitness(items, knapsack_max_capacity, individual)/f_j for individual in population ]
    chosen_probs = []
    for n in range(n_selection):
        r = random.random()
        cumulative_prob = 0.0
        for i, individual_prob in enumerate(probability):
            cumulative_prob += individual_prob
            if r <= cumulative_prob and i not in chosen_probs:
                chosen_probs.append(i)
                break
    chosen_individuals = [population[i] for i in chosen_probs]
    return chosen_individuals


def crossover(parents):
    children = []
    for i in range(0, len(parents) - 1 ,2):
        parent1 = parents[i]
        parent2 = parents[i+1]
        crossover_point = random.randint(1, len(parent1) - 2)
        children.append(parent1[:crossover_point] + parent2[crossover_point:])
        children.append(parent2[:crossover_point] + parent1[crossover_point:])
    return children


def get_elite(n_elite):
    selected_with_fitness = [(individual, fitness(items, knapsack_max_capacity, individual)) for individual in population]
    sorted_population = sorted(selected_with_fitness, key=lambda x: x[1], reverse=True)
    elite_individuals = [individual for individual, _ in sorted_population[:n_elite]]
    return elite_individuals


def mutation(children, p_Mut):
    new_children = []
    for child in children:
        for i in range(len(child) - 1):
            if random.uniform(0, 1) < p_Mut:
                child[i] = not child[i]
        new_children.append(child)
    return new_children


for _ in range(generations):
    population_history.append(population)
    # TODO: implement genetic algorithm
    parents = roulette_wheel_selection(population, n_selection)

    next_gen = get_elite(n_elite)
    children = crossover(parents)

    children = mutation(children, p_Mut)
    next_gen += children
    remaining_size = population_size - n_elite - n_selection
    next_gen += roulette_wheel_selection(population, remaining_size)
    population = next_gen

    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 100
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
