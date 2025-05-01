import math
import heapq
import numpy as np
from collections import defaultdict

''' This is most useful when you put delta crazy high, like 0.9.
The only thing you want out of this is the most optimistic sink paths. (which is a set).
You should let an LLM choose from that set. '''


class GraphCUCBPlanner:
    def __init__(self, delta=None, force_total_rounds=None):
        # adjacency list: node -> list of (neighbor, edge_id)
        self.adj = defaultdict(list)
        self.edge_stats = {}  # edge_id -> {'success': int, 'failure': int}
        self.edge_map = {}    # (u, v) -> edge_id
        self.total_rounds = 0
        self.delta = delta
        self.force_total_rounds = force_total_rounds

    def sink_nodes_set(self):
        """Return the set of nodes with no outgoing edges."""
        all_nodes = set()
        nodes_with_outgoing = set()

        for node, neighbors in self.adj.items():
            all_nodes.add(node)
            for neighbor, _ in neighbors:
                all_nodes.add(neighbor)
                nodes_with_outgoing.add(node)
        return all_nodes - nodes_with_outgoing

    def extend_path(self, path):
        """Return all possible extensions of the given path."""
        last_node = path[-1]
        neighbors = self.adj[last_node]
        return [path + [neighbor] for neighbor, _ in neighbors]

    def all_sink_paths(self, start_node):
        sink_nodes = self.sink_nodes_set()
        worklist = [[start_node]]
        all_sink_paths = []
        while worklist:
            path = worklist.pop(0)
            if path[-1] in sink_nodes:
                all_sink_paths.append(path)
            else:
                worklist.extend(self.extend_path(path))
        return all_sink_paths

    def add_edge(self, u, v, edge_id):
        """Add a directed edge from u to v with given edge_id."""
        self.adj[u].append((v, edge_id))
        self.edge_stats[edge_id] = {'success': 0, 'failure': 0}
        self.edge_map[(u, v)] = edge_id

    def record_result(self, edge_id, success):
        """Update the success/failure counts for an edge."""
        self.total_rounds += 1
        if success:
            self.edge_stats[edge_id]['success'] += 1
        else:
            self.edge_stats[edge_id]['failure'] += 1

    def set_results(self, results_dict):
        ''' results_dict is a dictionary of edge_id -> (success, failure)'''
        self.total_rounds = 0
        for edge_id, (success, failure) in results_dict.items():
            self.edge_stats[edge_id]['success'] = success
            self.edge_stats[edge_id]['failure'] = failure
            self.total_rounds += success + failure

    def compute_ucb(self, edge_id, quiet=True):
        """Return the UCB estimate for the success rate of an edge."""
        delta = self.delta
        s = self.edge_stats[edge_id]['success']
        f = self.edge_stats[edge_id]['failure']
        n = s + f
        if not quiet:
            print(f"Edge {edge_id} stats: success={s}, failure={f}, total={n}")

        if n == 0:
            if not quiet:
                print(f"Edge {edge_id} unexplored, returning optimistic 1.0")
            return 1.0  # Maximum optimism for unexplored edges

        empirical_mean = s / n
        if not quiet:
            print(f"Edge {edge_id} empirical mean: {empirical_mean:.3f}")

        if delta is None:
            bonus = np.sqrt(2 * np.log(self.total_rounds) / n)
            if not quiet:
                print(
                    f"Edge {edge_id} default bonus: {bonus:.3f} (total_rounds={self.total_rounds})")
        else:
            if self.force_total_rounds is not None:
                bonus = math.sqrt(
                    2 * math.log(max(1, self.force_total_rounds) / delta) / n)
            else:
                bonus = math.sqrt(
                    2 * math.log(max(1, self.total_rounds) / delta) / n)
            if not quiet:
                print(
                    f"Edge {edge_id} custom bonus: {bonus:.3f} (delta={delta})")

        result = min(1.0, empirical_mean + bonus)
        if not quiet:
            print(f"Edge {edge_id} final UCB: {result:.3f}")
        return result

    def plan_with_ucb(self, start, goal):
        """Use Dijkstra with -log(UCB) edge weights to find optimistic path."""
        heap = [(0.0, start, [])]  # (cost, current_node, path)
        visited = set()

        while heap:
            cost, node, path = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)

            if node == goal:
                # convert back to estimated path success
                return path, math.exp(-cost)

            for neighbor, edge_id in self.adj[node]:
                ucb = self.compute_ucb(edge_id)
                edge_cost = -math.log(ucb + 1e-9)  # avoid log(0)
                heapq.heappush(
                    heap, (cost + edge_cost, neighbor, path + [edge_id]))

        return None, 0.0  # no path found

    def compute_path_estimated_path_success(self, path):
        cost, path_idx = 0.0, 0
        while path_idx < len(path) - 1:
            path_idx += 1
            next_edge = (path[path_idx-1], path[path_idx])
            if next_edge not in self.edge_map:
                raise ValueError(f"Edge {next_edge} not found in edge_map")
            edge_id = self.edge_map[next_edge]
            ucb = self.compute_ucb(edge_id)
            edge_cost = -math.log(ucb + 1e-9)
            cost += edge_cost
            # Now cost is the cost to get to path[path_idx]
        return math.exp(-cost)

    def compute_estimated_success_all_sink_paths(self, start):
        all_sink_paths = list(self.all_sink_paths(start))
        return [(path, self.compute_path_estimated_path_success(path)) for path in all_sink_paths]

    def most_optimistic_sink_paths(self, start, slack=0.01):
        path_success_pairs = self.compute_estimated_success_all_sink_paths(
            start)
        max_success = path_success_pairs[0][1]
        for path, success in path_success_pairs:
            if success >= max_success:
                max_success = success
        good_paths = []
        for path, success in path_success_pairs:
            if success >= max_success - slack:
                good_paths.append((path, success))
        return good_paths

    def print_edge_stats(self):
        print(f"Total rounds: {self.total_rounds}")
        for edge_id, stats in self.edge_stats.items():
            s, f = stats['success'], stats['failure']
            ucb = self.compute_ucb(edge_id)
            print(f"Edge {edge_id}: {s} success, {f} fail, UCB={ucb:.3f}")
