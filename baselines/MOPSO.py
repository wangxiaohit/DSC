import numpy as np
import json
import time
import math

from tqdm import tqdm
from src.GA import fitness
from src.requirement_matching import RequirementMatching, find_pareto_optimals


def sort_dict_by_weighted_dimension(models_dict, weights):
    max_weight_index = list(weights).index(max(weights))
    reverse = True if max_weight_index == 2 else False
    for key in models_dict:
        models_dict[key].sort(key=lambda x: x[max_weight_index], reverse=reverse)
    return models_dict


def MOPSO(models_dict, pop_size=100, MAX_Iter=500, weights=[0.25, 0.25, 0.25, 0.25]):
    c1, c2 = 2.5, 2.5
    sorted_dict = sort_dict_by_weighted_dimension(models_dict, weights)
    ranges = [len(value) for value in models_dict.values()]
    dimension = len(ranges)
    
    V = [np.random.randint(-10, 11, size=dimension) for _ in range(pop_size)]
    X = [[np.random.choice(length) for length in ranges] for _ in range(pop_size)]
    fitnesses = [fitness(individual, models_dict, weights) for individual in X]

    Pibest = X.copy()
    Pibest_fitness = fitnesses.copy()
    Pgbest_fitness = max(fitnesses)
    best_index = fitnesses.index(Pgbest_fitness)
    Pgbest = X[best_index]
    tpre = 100
    count = 0
    wmax = 0.9
    wmin = 0.4
    w = wmax
    for t in range(MAX_Iter):
        if count >= tpre:
            V = [np.random.randint(-10, 11, size=dimension) for _ in range(pop_size)]
            X = [[np.random.choice(length) for length in ranges] for _ in range(pop_size)]
            count = 0
        else:
            c1 *= 0.99
            c2 *= 0.99
            w = wmin + (wmax - wmin) * (MAX_Iter - t) / MAX_Iter
            for i in range(pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                V[i] = w * V[i] + c1 * r1 * (np.array(Pibest[i]) - np.array(X[i])) + c2 * r2 * (np.array(Pgbest) - np.array(X[i]))
                X[i] = np.round(X[i] + V[i]).astype(int) % ranges
        new_fitnesses = [fitness(individual, models_dict, weights) for individual in X]

        for i in range(len(X)):
            if new_fitnesses[i] > Pibest_fitness[i]:
                Pibest[i] = X[i]
                Pibest_fitness[i] = new_fitnesses[i]
        max_fitness = max(Pibest_fitness)
        if max_fitness > Pgbest_fitness:
            best_index = Pibest_fitness.index(max_fitness)
            Pgbest = Pibest[best_index]
            Pgbest_fitness = max_fitness
        
        count += 1
    
    return -Pgbest_fitness, Pgbest


class mopso:
    def __init__(self, requirements, services, service_rtp, service_node_index, pattern_dict, gnn_solution, label_to_req_sol, node):
        self.requirements = requirements
        self.services = services
        self.service_node_index = service_node_index
        self.label_to_req_sol = label_to_req_sol
        self.node = node

        self.self_interest_model = RequirementMatching(requirements, services, service_rtp, service_node_index, pattern_dict, gnn_solution, label_to_req_sol) 

    def objective_function(self, indicators, weights):
        arithmetic_means = np.mean(indicators, axis=0)[[0, 1, 3]]
        geometric_mean = np.exp(np.mean(np.log(np.maximum(indicators, 1e-10)), axis=0)[2])
        weighted_arithmetic_means = arithmetic_means * weights[[0, 1, 3]]
        weighted_geometric_mean = geometric_mean * weights[2]
        return sum(weighted_arithmetic_means) - weighted_geometric_mean

    def sort_lists(self, lists, weights=np.array([0.25, 0.25, 0.25, 0.25], dtype=float)):
        objectives = [self.objective_function(np.array([indicators]), weights) for indicators in lists]
        sorted_indices = sorted(range(len(lists)), key=lambda i: objectives[i])
        sorted_lists = [lists[i] for i in sorted_indices]
        return sorted_lists

    def start(self, normalized_best_solutions_data, is_test=True, better_if_lower=[True, True, False, True]):
        np.random.seed(7)
        service_keys = list(self.services.keys())
        file_path = f'solutions/sol_mopso.json'
        solutions = []
        for t, (req, nbsd) in tqdm(enumerate(zip(self.requirements, normalized_best_solutions_data))):
            if is_test and 0 <= t < round(len(self.requirements) * 0.85):
                continue
            services_dict = {f"model{i}": self.services[service_keys[req_item]] for i, req_item in enumerate(req["reqs"])}
            weights = np.array(req["weights"], dtype=float)
            pareto_optimals_dict = {model_type: find_pareto_optimals(models, better_if_lower) for model_type, models in services_dict.items()}
            cannot_used_svcs_by_self_interest = self.self_interest_model.update_self_interest_degree()
            self.self_interest_model.matching_degree_module()

            new_pareto_optimals_dict = {}
            for r, (key, value) in zip(req["reqs"], pareto_optimals_dict.items()):
                new_pareto_optimals_dict[key] = [v for i, v in enumerate(value) if (r, i) not in cannot_used_svcs_by_self_interest]

            services_for_node = [[[] for _ in services_dict.keys()] for _ in range(self.node)]
            for r, (i, value) in zip(req["reqs"], enumerate(new_pareto_optimals_dict.values())):
                for j, v in enumerate(value):
                    n = self.service_node_index[(r, j)]
                    services_for_node[n][i].append(v)
            
            tt = time.time()
            new_pareto_optimals_dict_by_node = {f"model{i}": [] for i, req_item in enumerate(req["reqs"])}
            exposed_svcs = []
            for node_svcs in services_for_node:
                for i, svcs in enumerate(node_svcs):
                    sorted_svcs = self.sort_lists(svcs)
                    new_pareto_optimals_dict_by_node[f"model{i}"] += sorted_svcs[:3]
                    exposed_svcs += sorted_svcs[:3]

            best_value, best_solution = MOPSO(new_pareto_optimals_dict_by_node, weights=weights)

            ori_best_solution = []
            for i, (r, svc_index) in enumerate(zip(req["reqs"], best_solution)):
                svc = new_pareto_optimals_dict_by_node[f"model{i}"][svc_index]
                ori_svc_index = pareto_optimals_dict[f"model{i}"].index(svc)
                ori_best_solution.append((r, ori_svc_index))
            solution_info = {
                "req_number": t,
                "best_value": nbsd / (best_value + 1),
                "best_solution": ori_best_solution,
                "exposed_svcs": len(exposed_svcs) / len(self.label_to_req_sol),
                "time": time.time() - tt,
                "k": 2
            }
            solutions.append(solution_info)
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(solutions, file, ensure_ascii=False, indent=4)
            
            for sol in ori_best_solution:
                self.self_interest_model.services_used_dict[sol].append(1)
            for key, value in self.self_interest_model.services_used_dict.items():
                if len(value) <= t - round(len(self.requirements) * 0.85):
                    self.self_interest_model.services_used_dict[key].append(0)
            
