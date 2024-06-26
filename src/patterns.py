from pyfpgrowth import pyfpgrowth
import time


def mine_patterns(best_solutions, requirements):
    connected_list = []
    best_solutions_train = best_solutions[: round(len(requirements) * 0.7)]
    requirements_train = requirements[: round(len(requirements) * 0.7)]
    for s, r in zip(best_solutions_train, requirements_train):
        bs = s['best_solution']
        reqs = r['reqs']
        connected_sublist = [f"{req}-{sol}" for req, sol in zip(reqs, bs)]
        connected_list.append(connected_sublist)

    min_support = round(len(requirements) * 0.7 // 50)
    patterns = pyfpgrowth.find_frequent_patterns(connected_list, min_support)
    result_dict = {}
    for item, support in patterns.items():
        if len(item) >= 2:
            keys = tuple(int(x.split('-')[0]) for x in item)
            values = tuple(int(x.split('-')[1]) for x in item)
            if keys not in result_dict:
                result_dict[keys] = [[values, support]]
            else:
                result_dict[keys].append([values, support])
    
    return result_dict


def extract_corresponding_numbers(result_dict):
    new_dict = {num: list(set(val[i] for val in value_list))
                for key, value_list in result_dict.items()
                for i, num in enumerate(key)}
    return new_dict


def cover_without_overlap(numbers, result_dict, pareto_optimals_dict, matching_degree, cannot_used_svcs, services_e, prob_svc):

    def sort_key(key):
        length = len(key)
        degree_sum = sum(matching_degree.get(key, []))
        return (-degree_sum, -length)

    new_result_dict = {}
    new_matching_degree = {}

    for rp, sps in result_dict.items():
        valid_sps = [sp for i, sp in enumerate(sps) if all((r, s) not in cannot_used_svcs and services_e[(r, s)] > prob_svc for r, s in zip(rp, sp[0]))]
        if valid_sps:
            new_result_dict[rp] = valid_sps
            new_matching_degree[rp] = [matching_degree[rp][i] for i, sp in enumerate(sps) if sp in valid_sps]
    
    covered_numbers = set()
    used_keys = []

    sorted_keys = sorted(new_result_dict.keys(), key=sort_key)

    for key in sorted_keys:
        if set(key).issubset(numbers):
            indexes = []
            for num in key:
                index = numbers.index(num)
                if index not in covered_numbers:
                    indexes.append(index)
            if len(indexes) == len(key):
                used_keys.append(key)
                covered_numbers.update(indexes)

    covered_indexes = sorted(list(covered_numbers))
    covered_numbers = [numbers[i] for i in covered_indexes]
    
    req_to_svcs = {key: [lst[0] for lst in new_result_dict[key]] for key in used_keys}
    new_req_to_svcs = extract_corresponding_numbers(req_to_svcs)

    pareto_optimals_dict_keys = list(pareto_optimals_dict.keys())
    pareto_optimals_dict_new = dict()
    for req, svcs in new_req_to_svcs.items():
        index = numbers.index(req)
        pareto_optimals_dict_new[pareto_optimals_dict_keys[index]] = [pareto_optimals_dict[pareto_optimals_dict_keys[index]][svc] for svc in svcs]
    
    return pareto_optimals_dict_new, covered_numbers