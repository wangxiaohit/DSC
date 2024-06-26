import time
import numpy as np
import json

from collections import defaultdict
from itertools import chain, product
from tqdm import tqdm
from src.requirement_matching import RequirementMatching, find_pareto_optimals


class sboti:
    def __init__(self, requirements, services, service_rtp, service_node_index, pattern_dict, gnn_solution, label_to_req_sol, node):
        self.requirements = requirements
        self.services = services
        self.service_node_index = service_node_index
        self.label_to_req_sol = label_to_req_sol
        self.node = node

        self.self_interest_model = RequirementMatching(requirements, services, service_rtp, service_node_index, pattern_dict, gnn_solution, label_to_req_sol) 

        self.T = 4
        self.evaporation_rate = 0.015
        self.Q1 = 55

    def objective_function(self, indicators, weights):
        arithmetic_means = np.mean(indicators, axis=0)[[0, 1, 3]]
        geometric_mean = np.exp(np.mean(np.log(np.maximum(indicators, 1e-10)), axis=0)[2])
        weighted_arithmetic_means = arithmetic_means * weights[[0, 1, 3]]
        weighted_geometric_mean = geometric_mean * weights[2]
        return -(sum(weighted_arithmetic_means) - weighted_geometric_mean)

    def select_next_service(self, current_node, current_service, current_service_index, services_for_node, pheromone_store):
        next_services = services_for_node[current_node][current_service + 1]
        pheromones = np.array([pheromone_store[(current_node, current_service, current_service_index, next_service_index)]
                              for next_service_index in range(len(next_services))])
        probabilities = pheromones / np.sum(pheromones)
        next_service_index = np.random.choice(range(len(next_services)), p=probabilities)
        return next_service_index

    def generate_service_solution(self, start_node, services_for_node, pheromone_store, weights):
        service_solution = []
        service_solution_index = []
        current_node = start_node
        current_service = -1
        next_service_index = 0
        
        while len(service_solution) < len(services_for_node[current_node]):
            next_service_index = self.select_next_service(current_node, current_service, next_service_index, services_for_node, pheromone_store)
            current_service += 1
            service_solution.append(services_for_node[current_node][current_service][next_service_index])
            service_solution_index.append(next_service_index)

        return service_solution, service_solution_index, self.objective_function(service_solution, weights)

    def find_max_value(self, matrix):
        max_value = float('-inf')
        max_row = -1
        max_col = -1
        
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] > max_value:
                    max_value = matrix[i][j]
                    max_row = i
                    max_col = j
        
        return max_value, (max_row, max_col)

    def start(self, normalized_best_solutions_data, is_test=True, better_if_lower=[True, True, False, True]):
        np.random.seed(7)
        service_keys = list(self.services.keys())
        file_path = f'solutions/sol_sboti.json'
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
            init_mobile_agent = [0 for _ in range(self.node)]
            for r, (i, value) in zip(req["reqs"], enumerate(new_pareto_optimals_dict.values())):
                for j, v in enumerate(value):
                    n = self.service_node_index[(r, j)]
                    init_mobile_agent[n] += 1
                    services_for_node[n][i].append(v)
            
            DEFAULT_SERVICE = [1, 1, 0, 1]

            can_not_construct_sols_nodes = {i for i, node_svcs in enumerate(services_for_node) if any(len(svcs) == 0 for svcs in node_svcs)}

            no_used_svcs = defaultdict(list)
            for i in can_not_construct_sols_nodes:
                for j, svcs in enumerate(services_for_node[i]):
                    no_used_svcs[j].extend(svcs)
                    services_for_node[i][j] = [DEFAULT_SERVICE]
                init_mobile_agent[i] = len(services_for_node[i]) + 1

            if can_not_construct_sols_nodes:
                can_construct_sols_nodes = list(set(range(self.node)) - can_not_construct_sols_nodes)
                for i in can_construct_sols_nodes:
                    for j, svcs in enumerate(services_for_node[i]):
                        svcs.extend(no_used_svcs[j])
                        init_mobile_agent[i] += len(no_used_svcs[j])
                    break

            pheromone_store = {
                (i, -1, 0, k): 250
                for i in range(self.node)
                for k in range(len(services_for_node[i][0]))
            }
            pheromone_store.update({
                (i, j, k, l): 250
                for i in range(self.node)
                for j in range(len(services_for_node[i]) - 1)
                for k, l in product(range(len(services_for_node[i][j])), range(len(services_for_node[i][j+1])))
            })
            
            # start
            tt = time.time()
            iteration = 0
            best_value = float('-inf')
            best_solution = []
            exposed_svcs = set()
            k = 0
            while iteration < 100:
                results = [
                    [
                        self.generate_service_solution(start_node, services_for_node, pheromone_store, weights)
                        for _ in range(init_mobile_agent[start_node])
                    ]
                    for start_node in range(self.node)
                ]
                svc_sols = [[result[0] for result in node_results] for node_results in results]
                svc_sols_idxs = [[result[1] for result in node_results] for node_results in results]
                fitnesses = [[result[2] for result in node_results] for node_results in results]

                exp_fitness = [[np.exp(result[2]) for result in node_results] for node_results in results]

                for node_idx, (node_svc_sols_idxs, node_fitnesses) in enumerate(zip(svc_sols_idxs, exp_fitness)):  # 沉积
                    for svc_sol_idxs, fitness in zip(node_svc_sols_idxs, node_fitnesses):
                        pheromone_store[(node_idx, -1, 0, svc_sol_idxs[0])] += self.Q1 * fitness
                        for i in range(len(svc_sol_idxs) - 1):
                            pheromone_store[(node_idx, i, svc_sol_idxs[i], svc_sol_idxs[i+1])] += self.Q1 * fitness
                if iteration % self.T == 0:  # 蒸发
                    for key in pheromone_store:
                        pheromone_store[key] *= (1 - self.evaporation_rate)

                max_values = [max(node_fitnesses) for node_fitnesses in fitnesses]
                max_values_idx = [node_fitnesses.index(max(node_fitnesses)) for node_fitnesses in fitnesses]

                for i, (node_svc_sols, max_idx, max_value) in enumerate(zip(svc_sols_idxs, max_values_idx, max_values)):
                    if max_value > best_value:
                        for j, svc_idx in enumerate(node_svc_sols[max_idx]):
                            exposed_svcs.add((i, j, svc_idx))

                if max(max_values) > best_value:
                    node = max_values.index(max(max_values))
                    svc_idx = max_values_idx[max_values.index(max(max_values))]
                    best_value = max(max_values)
                    best_solution = svc_sols[node][svc_idx]
                    k += 1
                iteration += 1
            ori_best_solution = []
    
            for i, s in enumerate(best_solution):
                ori_index = pareto_optimals_dict[f"model{i}"].index(list(s))
                ori_best_solution.append((req["reqs"][i], ori_index))

            solution_info = {
                "req_number": t,
                "best_value": nbsd / (-best_value + 1),
                "best_solution": ori_best_solution,
                "exposed_svcs": len(exposed_svcs) / len(self.label_to_req_sol),
                "time": (time.time() - tt) / self.node,
                "k": k
            }
            solutions.append(solution_info)
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(solutions, file, ensure_ascii=False, indent=4)

            for sol in ori_best_solution:
                self.self_interest_model.services_used_dict[sol].append(1)
            for key, value in self.self_interest_model.services_used_dict.items():
                if len(value) <= t - round(len(self.requirements) * 0.85):
                    self.self_interest_model.services_used_dict[key].append(0)

            # services = [value for value in new_pareto_optimals_dict.values()]
            # num_ants = sum(len(value) for value in services)
            # pheromone_store = {
            #     (-1, 0, k): 250
            #     for k in range(len(services[0]))
            # }
            # pheromone_store.update({
            #     (j, k, l): 250
            #     for j in range(len(services) - 1)
            #     for k, l in product(range(len(services[j])), range(len(services[j+1])))
            # })
            # iteration = 0
            # best_value = 0

            # while iteration < 500:
            #     results = [self.generate_service_solution(services, pheromone_store, weights) for _ in range(num_ants)]
                
            #     svc_sols = [result[0] for result in results]
            #     svc_sols_idxs = [result[1] for result in results] 
            #     fitnesses = [result[2] for result in results]
            #     exp_fitness = [np.exp(fitness) for fitness in fitnesses]
                
            #     for svc_sol_idxs, fitness in zip(svc_sols_idxs, exp_fitness):
            #         for i in range(len(svc_sol_idxs) - 1):
            #             pheromone_store[(i, svc_sol_idxs[i], svc_sol_idxs[i+1])] += self.Q1 * fitness

            #     if iteration % self.T == 0:
            #         for key in pheromone_store:
            #             pheromone_store[key] *= (1 - self.evaporation_rate)
                        
            #     max_value, max_idx = max([(v,i) for i,v in enumerate(fitnesses)])
            #     if max_value > best_value:
            #         best_value = max_value
            #         best_solution = svc_sols_idxs[max_idx]

            #     iteration += 1

            # print(-best_value, best_solution)
            # time.sleep(100)