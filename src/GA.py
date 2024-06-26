import numpy as np
import random
import time


def objective_function(solution, models_dict, weights=[0.25, 0.25, 0.25, 0.25]):
    if len(weights) != 4:
        raise ValueError("weights must be a list of four elements.")

    weights = np.array(weights, dtype=float)

    indicators = [models_dict[f'model{i}'][index] for i, index in enumerate(solution)]
    arithmetic_means = np.mean(indicators, axis=0)[[0, 1, 3]]
    geometric_mean = np.exp(np.mean(np.log(np.maximum(indicators, 1e-10)), axis=0)[2])

    weighted_arithmetic_means = arithmetic_means * weights[[0, 1, 3]]
    weighted_geometric_mean = geometric_mean * weights[2]
    
    return sum(weighted_arithmetic_means) - weighted_geometric_mean


def fitness(solution, models_dict, weights):
    return -objective_function(solution, models_dict, weights)


def initialize_population(pop_size, models_dict):
    return [[random.choice(range(len(models_dict[model_type]))) for model_type in models_dict] for _ in range(pop_size)]


def selection(population, fitnesses, num_parents):
    parents = list(np.argsort(fitnesses)[-num_parents:])
    return [population[i] for i in parents]


def crossover(parents, offspring_size):
    offspring = []
    for _ in range(offspring_size):
        parent1, parent2 = random.sample(parents, 2)
        cross_point = random.randint(1, len(parent1) - 1)
        offspring.append(parent1[:cross_point] + parent2[cross_point:])
    return offspring


def mutate(offspring, models_dict, mutation_rate=0.1):
    for i in range(len(offspring)):
        if random.random() < mutation_rate:
            mutate_point = random.randint(0, len(offspring[i]) - 1)
            offspring[i][mutate_point] = random.randint(0, len(models_dict[f'model{mutate_point}']) - 1)
    return offspring


def genetic_algorithm(models_dict, pop_size=500, num_generations=100, num_parents=100, mutation_rate=0.1, optimal=True, weights=[0.25, 0.25, 0.25, 0.25]):
    if optimal:
        num_generations=500
    population = initialize_population(pop_size, models_dict)
    fitnesses = [fitness(individual, models_dict, weights) for individual in population]
    
    for generation in range(num_generations):
        parents = selection(population, fitnesses, num_parents)
        offspring_crossover = crossover(parents, offspring_size=pop_size - num_parents)
        offspring_mutation = mutate(offspring_crossover, models_dict, mutation_rate)
        population = parents + offspring_mutation
        fitnesses = [fitness(individual, models_dict, weights) for individual in population]

    best_index = np.argmax(fitnesses)
    return -max(fitnesses), population[best_index]
