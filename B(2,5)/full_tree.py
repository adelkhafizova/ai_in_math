import collections


class CleanTreeSolver:
    def __init__(self):
        self.rules = []
        self.visited_depths = {}
        self.parents = collections.defaultdict(set)
        self.actions = {}

    def add_rule(self, pattern, replacement):
        self.rules.append((pattern, replacement))

    def add_equivalence(self, p1, p2):
        self.rules.append((p1, p2))
        self.rules.append((p2, p1))

    def _get_transitions(self, s):
        moves = []
        for pat, repl in self.rules:
            start = 0
            while True:
                idx = s.find(pat, start)
                if idx == -1:
                    break
                new_s = s[:idx] + repl + s[idx + len(pat):]
                if repl == "":
                    desc = f"delete {pat}"
                else:
                    desc = f"replace {pat}->{repl}"
                moves.append((new_s, desc))
                start = idx + 1

        n = len(s)
        for length in range(1, n // 3 + 1):
            for i in range(0, n - 3 * length + 1):
                w = s[i:i + length]
                if s[i + length:i + 2 * length] == w and s[i + 2 * length:i + 3 * length] == w:
                    new_s = s[:i] + s[i + 3 * length:]
                    desc = f"delete ({w})^3"
                    moves.append((new_s, desc))

        return moves

    def solve(self, start_string, max_depth=None):
        queue = collections.deque([(start_string, 0)])
        self.visited_depths = {start_string: 0}
        self.parents.clear()
        self.actions.clear()

        print(f"Searching all winning paths for: '{start_string}'")

        found_win = False

        while queue:
            curr, depth = queue.popleft()

            if max_depth is not None and depth >= max_depth:
                continue

            for next_s, action in self._get_transitions(curr):
                if next_s == "":
                    found_win = True
                    self.parents[""].add(curr)
                    self.actions[(curr, "")] = action
                    continue

                if next_s not in self.visited_depths or self.visited_depths[next_s] == depth + 1:
                    if next_s not in self.visited_depths:
                        self.visited_depths[next_s] = depth + 1
                        queue.append((next_s, depth + 1))

                    self.parents[next_s].add(curr)
                    self.actions[(curr, next_s)] = action

        if not found_win:
            print("No solutions found.")
            return

        final_tree_nodes = set([""])
        q = collections.deque([""])

        while q:
            node = q.popleft()
            for p in self.parents[node]:
                if p not in final_tree_nodes:
                    final_tree_nodes.add(p)
                    q.append(p)

        print(f"\nThe solution tree contains {len(final_tree_nodes)} states.")

        return final_tree_nodes

    def print_tree_text(self, nodes):
        layers = collections.defaultdict(list)

        for node in nodes:
            if node == "":
                d = "WIN"
            else:
                d = self.visited_depths[node]
            layers[d].append(node)

        print("\nSolution Tree (layered):")

        sorted_keys = sorted(k for k in layers.keys() if isinstance(k, int))
        if "WIN" in layers:
            sorted_keys.append("WIN")

        for d in sorted_keys:
            print(f"\nLayer {d}:")
            for node in layers[d]:
                children = []
                for potential_child in nodes:
                    if node in self.parents[potential_child]:
                        act = self.actions[(node, potential_child)]
                        child_str = potential_child if potential_child != "" else "(EMPTY)"
                        children.append(f"   --[{act}]--> {child_str}")

                node_str = node if node != "" else "(EMPTY)"
                print(f" State: {node_str}")
                for c in children:
                    print(c)


if __name__ == "__main__":
    solver = CleanTreeSolver()
    solver.add_equivalence("abab", "bbaa")
    solver.add_equivalence("baba", "aabb")
    solver.add_equivalence("abba", "baab")

    target = "aabbabaabbabaaaabbaba"

    nodes = solver.solve(target)
    if nodes:
        solver.print_tree_text(nodes)