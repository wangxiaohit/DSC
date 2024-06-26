import time
import copy
import numpy as np
import math
import random
import json

from tqdm import tqdm
from src.requirement_matching import RequirementMatching, find_pareto_optimals


class TSP:
    def __init__(self, n, goalNode, constraint, distance, virtualNodeId, weights):
        self.n = n + 1
        self.virtualNodeId = virtualNodeId
        self.goalNode = goalNode
        self.constraint = constraint
        self.distance = distance
        self.weights = weights
        self.T0 = 30
        self.Tend = 1e-8
        self.L = 10
        self.a = 0.98

        for key in self.goalNode:
            for i in range(len(self.goalNode[key])):
                self.goalNode[key][i]["reqId"] = key

    def calc_best(self, rou):
        startPoint = 0
        for i in range(len(rou)):
            if rou[i]["nodeId"] == self.virtualNodeId:
                startPoint = i
                break
        newRou = rou[startPoint:] + rou[:startPoint]
        sumdis = 0.0
        for i in range(self.n - 1):
            sumdis += self.distance[newRou[i]["nodeId"]][newRou[i + 1]["nodeId"]] * i
        sumdis += self.distance[newRou[self.n - 1]["nodeId"]][newRou[0]["nodeId"]] * self.n - 1

        ma = [newRou[i]["QoS"] for i in range(len(newRou)) if len(newRou[i]["QoS"]) > 0]
        arithmetic_means = np.mean(ma, axis=0)[[0, 1, 3]]
        geometric_mean = np.exp(np.mean(np.log(np.maximum(ma, 1e-10)), axis=0)[2])
        weighted_arithmetic_means = arithmetic_means * self.weights[[0, 1, 3]]
        weighted_geometric_mean = geometric_mean * self.weights[2]
        objFuncValue = sum(weighted_arithmetic_means) - weighted_geometric_mean
        return sumdis + objFuncValue

    def getnewroute(self, route, times):
        current = copy.copy(route)

        if (times % 2 == 0) or (self.n <= 2):
            u = random.randint(0, self.n - 1)
            v = random.randint(0, self.n - 1)
            current[u] = random.choice(self.goalNode[current[u]["reqId"]])
            current[v] = random.choice(self.goalNode[current[v]["reqId"]])
            temp = current[u]
            current[u] = current[v]
            current[v] = temp
        else:
            temp2 = random.sample(range(0, self.n), 3)
            temp2.sort()
            u = temp2[0]
            v = temp2[1]
            w = temp2[2]
            w1 = w + 1
            temp3 = [0 for col in range(v - u + 1)]
            j = 0
            for i in range(u, v + 1):
                temp3[j] = current[i]
                j += 1

            for i2 in range(v + 1, w + 1):
                current[i2 - (v - u + 1)] = current[i2]
            w = w - (v - u + 1)
            j = 0
            for i3 in range(w + 1, w1):
                current[i3] = temp3[j]
                j += 1
        return current

    def solve(self):
        route = []
        for i in range(self.n):
            route.append(random.choice(self.goalNode[self.constraint[i]]))
        total_dis = self.calc_best(route)
        newroute = []
        new_total_dis = 0.0
        best = route
        best_total_dis = total_dis
        t = self.T0

        while True:
            if t <= self.Tend:
                break
            for rt2 in range(self.L):
                newroute = self.getnewroute(route, rt2)
                new_total_dis = self.calc_best(newroute)
                delt = new_total_dis - total_dis
                if delt <= 0:
                    route = newroute
                    total_dis = new_total_dis
                    if best_total_dis > new_total_dis:
                        best = newroute
                        best_total_dis = new_total_dis
                elif delt > 0:
                    p = math.exp(-delt / t)
                    ranp = random.uniform(0, 1)
                    if ranp < p:
                        route = newroute
                        total_dis = new_total_dis
            t = t * self.a
        return best, best_total_dis


class Node:
    def __init__(self, isRoot, children, parent, service, _lambda, constraint, weights, nodeId=0):
        self.nodeId = nodeId
        self.children = children
        self.parent = parent
        self.isRoot = isRoot
        self.weights = weights

        self.service = service
        for i in range(len(self.service)):
            self.service[i] = tuple(self.service[i])

        if len(self.service) == 1:
            self.service.append(self.service[0])

        self.contextFinished = []
        self._lambda = _lambda
        self.constraint = constraint
        self.u = dict()
        self.ud = dict()
        self.t = dict()
        self.td = dict()

        self.sa = set()
        self.bd = dict()
        self.b_ = dict()
        self.epsilon = -6
        self.delta = 0.1

        self.samplesum = 0
        self.bestp = -10000
        self.bestc = None

        if self.isRoot:
            self.parentFinished = True
        else:
            self.parentFinished = False

    def parent_start(self):
        d = self.sample([])
        context = [d]
        return self.children, context, "send_child_c"

    def sample(self, context):
        ct = tuple(context)
        if ct not in self.t:
            self.t[ct] = 1
        else:
            self.t[ct] += 1
        
        jud = 1
        while jud and self.t[ct] <= len(self.service):
            s = self.service[self.t[ct] - 1]
            ctd = tuple(context + [s])
            jud = 0
            if jud:
                self.td[ctd] = -1
                self.t[ct] += 1
            else:
                self.td[ctd] = 1

        if self.t[ct] <= len(self.service):
            d = self.service[self.t[ct] - 1]
        else:
            xk = -10000
            d = None
            for s in self.service:
                ctd = tuple(context + [s])
                if s not in self.sa and self.td[ctd] != -1:
                    if self.bd[ctd] > xk:
                        xk = self.bd[ctd]
                        d = s
                if d is None:
                    if self.td[ctd] != -1:
                        if self.bd[ctd] > xk:
                            xk = self.bd[ctd]
                            d = s
            ctd = tuple(context + [d])
            if d:
                self.td[ctd] += 1
        return d


    def min_d(self, context):
        maxp = float('-inf')
        maxs = None
        for s in self.service:
            indicators = context + [s]
            arithmetic_means = np.mean(indicators, axis=0)[[0, 1, 3]]
            geometric_mean = np.exp(np.mean(np.log(np.maximum(indicators, 1e-10)), axis=0)[2])
            weighted_arithmetic_means = arithmetic_means * self.weights[[0, 1, 3]]
            weighted_geometric_mean = geometric_mean * self.weights[2]
            p = sum(weighted_arithmetic_means) - weighted_geometric_mean
            if 1 - p > maxp:
                maxp = 1 - p
                maxs = s
        return maxp, maxs

    def is_satisfied(self, context, satisfied=False):
        ct = tuple(context)
        ep = 10000

        if self.u[ct] == -10000:
            return False

        if self.t[ct] <= len(self.service):
            return False

        for s in self.service:
            ctd = tuple(context + [s])
            if s not in self.sa and self.td[ctd] != -1:
                if self.u[ct] - (self.ud[ctd] + math.sqrt(math.log(2/self.delta) / self.td[ctd])) < ep:
                    ep = self.u[ct] - (self.ud[ctd] + math.sqrt(math.log(2/self.delta) / self.td[ctd]))
                    self.bestc = context + [s]

        if ep >= self.epsilon or satisfied:
            for ctd in self.ud.keys():
                if self.bestp < self.ud[ctd]:
                    self.bestp = self.ud[ctd]
                    self.bestc = list(ctd)
            return True
        else:
            return False

    def received_parent_c(self, context, satisfied=False):
        if len(self.children) == 0:
            l_min, d = self.min_d(context)
            if d:
                self.bd[tuple(context + [d])] = l_min
                return self.parent, (l_min, context, l_min), "send_parent_c"
            else:
                return self.parent, (-10000, context, -10000), "send_parent_c"
        else:
            d = self.sample(context)
            if d:
                return self.children, context + [d], "send_child_c"
            else:
                return self.parent, (-10000, context, -10000), "send_parent_c"

    def received_parent_f(self, context, satisfied=False):
        self.parentFinished = True
        self.contextFinished = context

        self.samplesum = 0

        if len(self.children) == 0:
            l_min, d = self.min_d(context)
            return self.children, (1 - l_min, context + [d]), "finish"
        elif self.is_satisfied(context, satisfied):
            return self.children, self.bestc, "send_child_f"
        else:
            d = self.sample(context)
            return self.children, context + [d], "send_child_c"

    def received_child_c(self, context, satisfied=False):
        ctd = tuple(context[1])
        ct = tuple(context[1][:-1])

        jud = 0
        times = 0

        if ct not in self.u:
            self.u[ct] = context[2] - jud
        else:
            self.u[ct] = max(context[2] - jud, self.u[ct])
        if ctd not in self.ud:
            self.ud[ctd] = context[2] - jud
        else:
            self.ud[ctd] = max(context[2] - jud, self.ud[ctd])

        self.b_[ctd] = context[0]

        self.sa = set()
        b = -10000
        for d in self.service:
            _ctd = tuple(context[1][:-1] + [d])

            if _ctd not in self.b_ or self.td[_ctd] == -1:
                continue
            ld = math.sqrt(2 * self._lambda * math.log(self.t[ct]) / self.td[_ctd])

            self.bd[_ctd] = max(self.ud[_ctd] + ld, self.b_[_ctd] - jud)

            b = max(b, self.bd[_ctd])
            if abs(self.bd[_ctd] - self.ud[_ctd]) < 0.01:
                self.sa.add(d)
        if self.parentFinished and self.is_satisfied(context[1][:-1], satisfied):
            return self.children, self.bestc, "send_child_f"
        elif self.parentFinished:
            d = self.sample(self.contextFinished)
            return self.children, self.contextFinished + [d], "send_child_c"
        else:
            return self.parent, (b, context[1][:-1], context[2] - jud), "send_parent_c"


class TSDSC:
    def __init__(self, services, routes, minpath, weights):
        self.services = services
        self.routes = routes
        self.minpath = minpath
        self.nodeNumber = 0
        self.weights = weights
        self.x = time.time()

    def findbest(self, services):
        bestService = None
        bestPrice = 0x77777777
        for service in services:
            if service[0] + 1 - service[1] < bestPrice:
                bestService = service
                bestPrice = service[0] + 1 - service[1]
        return bestService

    def requirementBroadcast(self):
        delay = 0
        delayTime = 0
        nodeMsglist = []
        visited = [0 for _ in range(len(self.routes))]
        visited[self.nodeNumber] = 1
        queue = [self.nodeNumber]
        while len(queue) > 0:
            node = queue.pop(0)
            services = self.services[node]
            nodeMsg = {
                "nodeNumber": node,
                "bestServices": []
            }
            for s in services:
                if len(s) > 0:
                    nodeMsg["bestServices"].append(self.findbest(s))
                else:
                    nodeMsg["bestServices"].append([])
            nodeMsglist.append(nodeMsg)
            for i in range(len(self.routes)):
                if self.routes[node][i] != 0 and not visited[i]:
                    visited[i] = 1
                    delay += 1
                    delayTime += self.minpath[node][i]
                    queue.append(i)
        return nodeMsglist, delay, delayTime

    def nodeSelection(self, nodeMsglist):
        goalNode = dict()
        for i in range(len(self.services[0])):
            goalNode[i] = []
        for nodeMsg in nodeMsglist:
            nNumber = nodeMsg["nodeNumber"]
            for i in range(len(nodeMsg["bestServices"])):
                if len(nodeMsg["bestServices"][i]) > 0:
                    goalNode[i].append({
                        "nodeId": nNumber,
                        "QoS": nodeMsg["bestServices"][i],
                        "reqName": i
                    })
        goalNode[-1] = [{'nodeId': len(self.minpath), 'QoS': [], "reqName": ""}]
        shortestPath = copy.deepcopy(self.minpath)
        shortestPath = np.append(shortestPath, [[0 for _ in range(len(shortestPath))]], axis=0)
        shortestPath = np.append(shortestPath, [[0] for _ in range(len(shortestPath))], axis=1)
        tsp = TSP(len(self.services[0]), goalNode, list(goalNode.keys()), shortestPath, virtualNodeId=len(shortestPath) - 1, weights=self.weights)

        best, bestTotalDis = tsp.solve()
        virtualStart = 0
        for i in range(len(best)):
            if best[i]["nodeId"] == len(shortestPath) - 1:
                virtualStart = i
        best = best[virtualStart:] + best[0: virtualStart]
        return best

    def constraintCoordination(self, best, delay, delayTime):
        nodeDUCT = []
        for i in range(1, len(best)):
            reqName = best[i]["reqName"]
            nodeId = best[i]["nodeId"]
            if i == 1:
                isRoot = True
                parent = []
            else:
                isRoot = False
                parent = [i - 2]
            if i == len(best) - 1:
                children = []
                con = []
            else:
                children = [i]
                con = []
            nodeDUCT.append(Node(isRoot=isRoot, children=children, parent=parent,
                                 service=self.services[best[i]["nodeId"]][reqName],
                                 _lambda=len(best) - 1 - i, constraint=con, nodeId=nodeId, weights=self.weights))

        finish = False
        x = time.time()
        children, c, messageType = nodeDUCT[0].parent_start()
        number = 0
        numberTime = 0
        satisfied = False
        exposed_svcs = set()
        while not finish:
            if time.time() - x > 10:
                return []
            oldNumberTime = numberTime
            if messageType == "send_child_c":
                for child in children:
                    numberTime += self.minpath[nodeDUCT[child - 1].nodeId][nodeDUCT[child].nodeId]
                    children, c, messageType = nodeDUCT[child].received_parent_c(c, satisfied)
            elif messageType == "send_parent_c":
                for child in children:
                    numberTime += self.minpath[nodeDUCT[child + 1].nodeId][nodeDUCT[child].nodeId]
                    children, c, messageType = nodeDUCT[child].received_child_c(c, satisfied)
            elif messageType == "send_child_f":
                for child in children:
                    numberTime += self.minpath[nodeDUCT[child - 1].nodeId][nodeDUCT[child].nodeId]
                    children, c, messageType = nodeDUCT[child].received_parent_f(c, satisfied)
            elif messageType == "finish":
                # print(c)
                finish = True
            if numberTime != oldNumberTime:
                number += 1
            
            exposed_svcs_c = c[1] if isinstance(c, tuple) and len(c) >= 2 else c
            exposed_svcs.update(exposed_svcs_c if isinstance(exposed_svcs_c, (list, set, tuple)) else [exposed_svcs_c])

        solution = {
            "algorName": "DUCT",
            "bestSolution": c,
            "runningTime": time.time() - self.x,
            "cooTimes": delay + number,
            "cooTime": delayTime + numberTime,
            "exposed_svcs": exposed_svcs
        }
        return solution

    def start(self, seed=7):
        random.seed(seed)
        nodeMsglist, delay, delayTime = self.requirementBroadcast()
        best = self.nodeSelection(nodeMsglist)
        req_ids = [d['reqId'] for d in best if d['reqId'] != -1]
        solution = self.constraintCoordination(best, delay, delayTime)
        return solution, req_ids


class tsdsc:
    def __init__(self, requirements, services, service_rtp, service_node_index, pattern_dict, gnn_solution, label_to_req_sol, node):
        self.requirements = requirements
        self.services = services
        self.service_node_index = service_node_index
        self.label_to_req_sol = label_to_req_sol
        self.node = node

        self.self_interest_model = RequirementMatching(requirements, services, service_rtp, service_node_index, pattern_dict, gnn_solution, label_to_req_sol) 

    def generate_matrices(self, n):
        routes = []
        for i in range(n):
            row = [1] * n
            row[i] = 0
            routes.append(row)

        minpath = []
        for i in range(n):
            row = [0.001] * n
            row[i] = 0
            minpath.append(row)

        return routes, minpath

    def start(self, normalized_best_solutions_data, is_test=True, better_if_lower=[True, True, False, True]):
        service_keys = list(self.services.keys())
        file_path = f'solutions/sol_tsdsc.json'
        solutions = []
        routes, minpath = self.generate_matrices(self.node)

        for t, (req, nbsd) in tqdm(enumerate(zip(self.requirements, normalized_best_solutions_data))):
            if is_test and 0 <= t < round(len(self.requirements) * 0.85):
                continue
            services_dict = {f"model{i}": self.services[service_keys[req_item]] for i, req_item in enumerate(req["reqs"])}
            weights = req["weights"]
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

            weights = np.array(req["weights"], dtype=float)
            model = TSDSC(services_for_node, routes, minpath, weights)
            solution, req_ids = model.start()
            
            ori_best_solution = []
            for req_id, s in zip(req_ids, solution["bestSolution"][1]):
                ori_index = pareto_optimals_dict[f"model{req_id}"].index(list(s))
                ori_best_solution.append((req["reqs"][req_id], ori_index))
            solution_info = {
                "req_number": t,
                "best_value": nbsd / (solution["bestSolution"][0] + 1),
                "best_solution": ori_best_solution,
                "exposed_svcs": len(solution["exposed_svcs"]) / len(self.label_to_req_sol),
                "time": solution["runningTime"],
                "k": solution["cooTimes"]
            }
            solutions.append(solution_info)
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(solutions, file, ensure_ascii=False, indent=4)

            for sol in ori_best_solution:
                self.self_interest_model.services_used_dict[sol].append(1)
            for key, value in self.self_interest_model.services_used_dict.items():
                if len(value) <= t - round(len(self.requirements) * 0.85):
                    self.self_interest_model.services_used_dict[key].append(0)
                    