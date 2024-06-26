import configparser
import json
import sys
import ast

from src.patterns import mine_patterns
from src.requirement_matching import RequirementMatching, find_pareto_optimals
from baselines.TSDSC import tsdsc
from baselines.SBOTI import sboti
from baselines.CSC import csc
from baselines.CSSA import cssa
from baselines.MOPSO import mopso


def convert_str_to_tuple(key):
    try:
        return ast.literal_eval(key)
    except (ValueError, SyntaxError):
        return key


def convert_keys_from_str(obj):
    if isinstance(obj, dict):
        return {convert_str_to_tuple(key): convert_keys_from_str(value) 
                for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_from_str(item) for item in obj]
    return obj


if __name__ == "__main__":
    config = configparser.RawConfigParser()
    config.read("environment.ini")
    dataset, approach = sys.argv[1], sys.argv[2]
    parakey = config.options(dataset)
    paravalue = [config.get(dataset, key) for key in parakey]
    node = int(paravalue[0])

    with open(f'data/services.json', 'r') as file:
        services = json.load(file)
    with open(f'data/requirements.json', 'r') as file:
        requirements_all = json.load(file)
        requirements_train = requirements_all[: 7000] + requirements_all[10000: 13500]
        requirements_test = requirements_all[8500: 10000] + requirements_all[14250:]
        requirements_val = requirements_all[7000: 8500] + requirements_all[13500: 14250]
        requirements_all = requirements_train + requirements_val + requirements_test
    with open(f'data/best_solutions.json', 'r') as file:
        best_solutions_all = json.load(file)
        best_solutions_test = best_solutions_all[8500: 10000] + best_solutions_all[14250:]
        best_solutions_val = best_solutions_all[7000: 8500] + best_solutions_all[13500: 14250]
        best_solutions_train = best_solutions_all[: 7000] + best_solutions_all[10000: 13500]
        best_solutions_all = best_solutions_train + best_solutions_val + best_solutions_test
        
        normalized_best_solutions_data = []
        for item in best_solutions_all:
            normalized_best_solutions_data.append(item["best_value"] + 1)

    better_if_lower = [True, True, False, True]
    service_keys = list(services.keys())
    solutions = []
    pareto_optimals_dict = {model_type: find_pareto_optimals(models, better_if_lower) for model_type, models in services.items()}
    label_to_req_sol = [(r, b) for r, key in enumerate(pareto_optimals_dict.keys()) for b, _ in enumerate(pareto_optimals_dict[key])]

    with open('data/service_rtp.json', 'r') as f:
        service_rtp = convert_keys_from_str(json.load(f))
    with open('data/service_node_index.json', 'r') as f:
        service_node_index = convert_keys_from_str(json.load(f))

    pattern_dict = mine_patterns(best_solutions_all, requirements_all)
    with open(f"solutions/gnn_solution_0.json", "r") as file:
        gnn_solution = json.load(file)
    if approach == "Ours":
        prob_svc, T, gnn_num, gnn_threshold = float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
        with open(f"solutions/gnn_solution_{gnn_num}.json", "r") as file:
            gnn_solution = json.load(file)
        model = RequirementMatching(requirements_all, services, service_rtp, service_node_index, pattern_dict, gnn_solution, label_to_req_sol, prob_svc, T, gnn_num, gnn_threshold)
        model.start(normalized_best_solutions_data)
    elif approach == "TSDSC":
        model = tsdsc(requirements_all, services, service_rtp, service_node_index, pattern_dict, gnn_solution, label_to_req_sol, node)
        model.start(normalized_best_solutions_data)
    elif approach == "CSC":
        model = csc(requirements_all, services, service_rtp, service_node_index, pattern_dict, gnn_solution, label_to_req_sol, node)
        model.start(normalized_best_solutions_data)
    elif approach == "SBOTI":
        model = sboti(requirements_all, services, service_rtp, service_node_index, pattern_dict, gnn_solution, label_to_req_sol, node)
        model.start(normalized_best_solutions_data)
    elif approach == "CSSA":
        model = cssa(requirements_all, services, service_rtp, service_node_index, pattern_dict, gnn_solution, label_to_req_sol, node)
        model.start(normalized_best_solutions_data)
    elif approach == "MOPSO":
        model = mopso(requirements_all, services, service_rtp, service_node_index, pattern_dict, gnn_solution, label_to_req_sol, node)
        model.start(normalized_best_solutions_data)
    else:
        print("Error!")