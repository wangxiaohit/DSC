import numpy as np
import json
import time
import math

from tqdm import tqdm
from src.GA import fitness
from src.requirement_matching import RequirementMatching, find_pareto_optimals


def CSSA(models_dict, pop_size=100, MAX_Iter=500, weights=[0.25, 0.25, 0.25, 0.25]):
    mu = 4
    ch = np.random.rand(pop_size)
    ranges = [len(value) for value in models_dict.values()]
    population = [[np.random.choice(length) for length in ranges] for _ in range(pop_size)]
    fitnesses = [fitness(individual, models_dict, weights) for individual in population]
    best_fitness = -max(fitnesses)
    best_pop = population[fitnesses.index(-best_fitness)]
    best_fitnesses, best_pops = [], []
    for t in range(MAX_Iter):
        c1, c2 = np.random.random(), np.random.random()
        ch = mu * ch * (1 - ch)
        for i in range(len(population)):
            if i <= len(population) / 2:
                _pop = np.array(population[i])
                j = np.random.choice(len(ranges))
                if c2 >= 0.5:
                    _pop[j] = best_pop[j] + np.floor(c1 * ranges[j])
                else:
                    _pop[j] = best_pop[j] - np.floor(c1 * ranges[j])
            else:
                _pop = np.array(best_pop)
                j = np.random.choice(len(ranges))
                _pop[j] = np.floor(ch[i] * ranges[j])
            population[i] = _pop % ranges
        fitnesses = [fitness(individual, models_dict, weights) for individual in population]
        best_fitness = -max(fitnesses)
        best_pop = population[fitnesses.index(-best_fitness)]
        best_fitnesses.append(best_fitness)
        best_pops.append(best_pop)
    
    return min(best_fitnesses), best_pops[best_fitnesses.index(min(best_fitnesses))]


class cssa:
    def __init__(self, requirements, services, service_rtp, service_node_index, pattern_dict, gnn_solution, label_to_req_sol, node):
        self.requirements = requirements
        self.services = services
        self.service_node_index = service_node_index
        self.label_to_req_sol = label_to_req_sol
        self.node = node

        self.self_interest_model = RequirementMatching(requirements, services, service_rtp, service_node_index, pattern_dict, gnn_solution, label_to_req_sol) 
        
    def start(self, normalized_best_solutions_data, is_test=True, better_if_lower=[True, True, False, True]):
        np.random.seed(7)
        service_keys = list(self.services.keys())
        file_path = f'solutions/sol_cssa.json'
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

            ti = time.time()
            best_value, best_solution = CSSA(new_pareto_optimals_dict, weights=weights)

            ori_best_solution = []
            for i, (r, svc_index) in enumerate(zip(req["reqs"], best_solution)):
                svc = new_pareto_optimals_dict[f"model{i}"][svc_index]
                ori_svc_index = pareto_optimals_dict[f"model{i}"].index(svc)
                ori_best_solution.append((r, ori_svc_index))

            solution_info = {
                "req_number": t,
                "best_value": nbsd / (best_value + 1),
                "best_solution": ori_best_solution,
                "time": time.time() - ti
            }
            solutions.append(solution_info)
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(solutions, file, ensure_ascii=False, indent=4)
            
            for sol in ori_best_solution:
                self.self_interest_model.services_used_dict[sol].append(1)
            for key, value in self.self_interest_model.services_used_dict.items():
                if len(value) <= t - round(len(self.requirements) * 0.85):
                    self.self_interest_model.services_used_dict[key].append(0)

