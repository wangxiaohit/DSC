import time
import numpy as np
import json

from tqdm import tqdm
from src.patterns import cover_without_overlap
from src.GA import genetic_algorithm


def dominates(model_a, model_b, criteria):
    better_in_any = False
    not_worse_in_all = True
    for a, b, lower_is_better in zip(model_a, model_b, criteria):
        if lower_is_better:
            if a > b:
                not_worse_in_all = False
            if a < b:
                better_in_any = True
        else:
            if a < b:
                not_worse_in_all = False
            if a > b:
                better_in_any = True

    return better_in_any and not_worse_in_all


def find_pareto_optimals(models, criteria):
    pareto_optimals = []
    for i, candidate in enumerate(models):
        is_dominated = False
        for j, competitor in enumerate(models):
            if i != j and dominates(competitor, candidate, criteria):
                is_dominated = True
                break
        if not is_dominated:
            pareto_optimals.append(candidate)
    return pareto_optimals


class RequirementMatching:
    def __init__(self, requirements, services, service_rtp, service_node_index, pattern_dict, gnn_solution, label_to_req_sol, prob_svc=0.3, T=20, gnn_num=0, gnn_threshold=5):
        self.requirements = requirements
        self.services = services
        self.service_rtp = service_rtp
        self.service_node_index = service_node_index
        self.pattern_dict = pattern_dict
        self.gnn_solution = gnn_solution
        self.label_to_req_sol = label_to_req_sol

        self.services_used_dict = {
            key: [] for key in self.service_node_index.keys()
        }
        self.services_e = {
            key: 0 for key in self.service_node_index.keys()
        }
        self.self_interest_degree = {
            key: 0 for key in self.service_node_index.keys()
        }
        self.matching_degree = {
            key: [0 for _ in values] for key, values in self.pattern_dict.items()
        }
        self.solutions = []

        self.T = T
        self.gnn_threshold = gnn_threshold
        self.prob_svc = prob_svc
        self.gnn_num = gnn_num


    def f(self, r, p, r_overline):
        res = 1 - np.power((1 - np.exp(-r * 0.05)) / (1 - np.exp(-r_overline * 0.05)), 2)
        profit = np.exp(-p)
        return np.add(res, profit) / 2

    def update_self_interest_degree(self):
        cannot_used_svcs_by_self_interest = set()
        for key, value in self.services_used_dict.items():
            svc_all_res = self.service_rtp[key][0]
            svc_time = self.service_rtp[key][1]
            svc_profit = self.service_rtp[key][2]
            svc_res = svc_all_res - np.sum(np.array(self.services_used_dict[key][-svc_time:]) == 1)
            self.self_interest_degree[key] = self.f(svc_res, svc_profit, svc_all_res)
            if self.self_interest_degree[key] > 0.5:
                cannot_used_svcs_by_self_interest.add(key)
        return cannot_used_svcs_by_self_interest

    def calc_e(self, used_list, tau=0):
        _tau = tau
        u_now = used_list[-self.T:]
        T = len(u_now)
        uj, vj = 0, 0
        t = 1
        for ui in u_now:
            if tau == 0:
                _tau = 1 - np.var(u_now)
            if ui == 1:
                uj += np.power(_tau, T-t)
            elif ui == -1:
                vj += np.power(_tau, T-t)
            t += 1
        e = (uj + 1)/(uj + vj + 2)
        return e

    def matching_degree_module(self):  # Calculate matching Degree
        for key, value in self.services_used_dict.items():
            self.services_e[key] = self.calc_e(value)

        for rp, sps in self.pattern_dict.items():
            degrees = []
            for sp in sps:
                svcs = [(r, s) for r, s in zip(rp, sp[0])]
                psp = np.cumprod([self.services_e[svc] for svc in svcs])[0]
                p = psp * np.log(sp[1])
                degrees.append(p)
            self.matching_degree[rp] = degrees

    def read_gnn_sol(self, req, sol, pareto_optimals_dict, cannot_used_svcs):
        threshold = 5
        pareto_optimals_new = {key: [] for key in pareto_optimals_dict}

        for label in sol:
            req_label, sol = self.label_to_req_sol[label]
            if req_label in req['reqs'] and self.services_e[(req_label, sol)] > self.prob_svc and (req_label, sol) not in cannot_used_svcs:
                model_key = f"model{req['reqs'].index(req_label)}"
                if len(pareto_optimals_new[model_key]) < threshold:
                    pareto_optimals_new[model_key].append(pareto_optimals_dict[model_key][sol])

        return pareto_optimals_new

    def start(self, normalized_best_solutions_data, is_test=True, better_if_lower=[True, True, False, True]):
        service_keys = list(self.services.keys())
        file_path = f'solutions/sol_{self.prob_svc}_{self.T}_{self.gnn_num}_{self.gnn_threshold}.json'
        solutions = []
        for t, (req, nbsd) in tqdm(enumerate(zip(self.requirements, normalized_best_solutions_data))):
            if is_test and 0 <= t < round(len(self.requirements) * 0.85):
                continue
            services_dict = {f"model{i}": self.services[service_keys[req_item]] for i, req_item in enumerate(req["reqs"])}
            weights = req["weights"]
            pareto_optimals_dict = {model_type: find_pareto_optimals(models, better_if_lower) for model_type, models in services_dict.items()}
            
            cannot_used_svcs_by_self_interest = self.update_self_interest_degree()
            self.matching_degree_module()

            exposed_svcs = set()
            cannot_used_svcs = set()
            ti = time.time()
            k = 1
            new_pareto_optimals_dict = dict()
            reserved_pareto_optimals_dict = dict()
            while k <= 10:
                pareto_optimals_dict_by_patterns, covered_numbers = cover_without_overlap(req["reqs"], self.pattern_dict, pareto_optimals_dict,
                                                                                          self.matching_degree, cannot_used_svcs, self.services_e,
                                                                                          self.prob_svc)
                pareto_optimals_dict_by_gnn = self.read_gnn_sol(req, self.gnn_solution[t], pareto_optimals_dict, cannot_used_svcs)
                for i, r in enumerate(req["reqs"]):
                    if f"model{i}" in reserved_pareto_optimals_dict:
                        new_pareto_optimals_dict[f"model{i}"] = reserved_pareto_optimals_dict[f"model{i}"]
                        continue
                    if r in set(covered_numbers):
                        pattern_len = len(pareto_optimals_dict_by_patterns[f"model{i}"])
                        new_pareto_optimals_dict[f"model{i}"] = pareto_optimals_dict_by_patterns[f"model{i}"]
                        if pattern_len < self.gnn_threshold:
                            new_pareto_optimals_dict[f"model{i}"] += pareto_optimals_dict_by_gnn[f"model{i}"][:self.gnn_threshold-pattern_len]
                    else:
                        new_pareto_optimals_dict[f"model{i}"] = pareto_optimals_dict_by_gnn[f"model{i}"]

                for values in new_pareto_optimals_dict.values():
                    for v in values:
                        exposed_svcs.add(tuple(v))

                best_value, best_solution = genetic_algorithm(new_pareto_optimals_dict, pop_size=200, weights=weights, optimal=False)

                all_available = True
                ori_best_solution = []
                reserved_pareto_optimals_dict = dict()
                for i, (r, svc_index) in enumerate(zip(req["reqs"], best_solution)):
                    svc = new_pareto_optimals_dict[f"model{i}"][svc_index]
                    ori_svc_index = pareto_optimals_dict[f"model{i}"].index(svc)
                    ori_best_solution.append((r, ori_svc_index))
                    if (r, ori_svc_index) in cannot_used_svcs_by_self_interest:
                        cannot_used_svcs.add((r, ori_svc_index))
                        all_available = False
                        self.services_used_dict[(r, ori_svc_index)].append(-1)
                    else:
                        reserved_pareto_optimals_dict[f"model{i}"] = [svc]
                        
                if all_available:
                    solution_info = {
                        "req_number": t,
                        "best_value": nbsd / (best_value + 1),
                        "best_solution": ori_best_solution,
                        "exposed_svcs": len(exposed_svcs) / len(self.label_to_req_sol),
                        "time": time.time() - ti,
                        "k": k
                    }
                    solutions.append(solution_info)
                    with open(file_path, 'w', encoding='utf-8') as file:
                        json.dump(solutions, file, ensure_ascii=False, indent=4)
                    
                    for sol in ori_best_solution:
                        self.services_used_dict[sol].append(1)
                    for key, value in self.services_used_dict.items():
                        if len(value) <= t - round(len(self.requirements) * 0.85):
                            self.services_used_dict[key].append(0)
                    break

                k += 1