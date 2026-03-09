import copy
from typing import Optional


# ── Tree data structure ────────────────────────────────────────────────────────

class TreeNode:
    """
    A node in an unordered rooted tree.
    Leaves have a non-None `name`; internal nodes have children.
    Children can be freely rotated (reordered) at any internal node.
    """
    def __init__(self, name: Optional[str] = None):
        self.name = name          # leaf label; None for internal nodes
        self.children: list["TreeNode"] = []

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def leaves(self) -> list[str]:
        """Return leaf labels in current left-to-right order."""
        if self.is_leaf():
            return [self.name]
        result = []
        for child in self.children:
            result.extend(child.leaves())
        return result

    def __repr__(self):
        if self.is_leaf():
            return f"Leaf({self.name!r})"
        return f"Node([{', '.join(repr(c) for c in self.children)}])"


# ── Crossing count ─────────────────────────────────────────────────────────────

def count_crossings(order_left: list[str], order_right: list[str]) -> int:
    """
    Count the number of crossing edges in a tanglegram layout.

    Leaves on the left are in `order_left` (top-to-bottom).
    Leaves on the right are in `order_right` (top-to-bottom).
    An edge (a, a) crosses edge (b, b) iff their relative vertical
    positions are inverted — i.e. this equals the number of inversions
    when the right-side positions are read in left-side order.

    Only shared leaves contribute crossings; unshared leaves are ignored.
    """
    shared = set(order_left) & set(order_right)
    pos_right = {name: i for i, name in enumerate(order_right) if name in shared}

    # Walk left order; collect right-side positions of shared leaves
    seq = [pos_right[name] for name in order_left if name in shared]

    # Count inversions via merge-sort — O(n log n)
    def merge_count(arr):
        if len(arr) <= 1:
            return arr, 0
        mid = len(arr) // 2
        left, lc = merge_count(arr[:mid])
        right, rc = merge_count(arr[mid:])
        merged, mc = [], 0
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i]); i += 1
            else:
                merged.append(right[j]); j += 1
                mc += len(left) - i      # all remaining left elements cross
        merged.extend(left[i:]); merged.extend(right[j:])
        return merged, lc + rc + mc

    _, crossings = merge_count(seq)
    return crossings


# ── One-sided optimisation ─────────────────────────────────────────────────────

def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def optimize_one_side(node: TreeNode, anchor_positions: dict[str, int]) -> float:
    """
    Recursively rotate `node`'s subtrees so that the median positions of
    their leaves (in the *other* tree's fixed ordering) are non-decreasing
    left-to-right.  This greedy step is optimal for binary trees and a good
    heuristic for higher-arity trees.

    Returns the median anchor position of this node's leaf set.
    """
    if node.is_leaf():
        return float(anchor_positions.get(node.name, 0))

    # Recurse and collect (median, child) pairs
    child_medians = []
    for child in node.children:
        med = optimize_one_side(child, anchor_positions)
        child_medians.append((med, child))

    # Sort children by ascending median → minimises crossings
    child_medians.sort(key=lambda x: x[0])
    node.children = [child for _, child in child_medians]

    # Return median of this node's leaves
    positions = [anchor_positions[lf] for lf in node.leaves() if lf in anchor_positions]
    return _median(positions) if positions else 0.0


# ── Two-sided iterative alignment ─────────────────────────────────────────────

def align_trees(
    tree1: TreeNode,
    tree2: TreeNode,
    max_iter: int = 20,
    inplace: bool = False,
) -> tuple[list[str], list[str], int]:
    """
    Iteratively optimise both trees to minimise tanglegram crossings.

    At each iteration:
      1. Fix tree1's current leaf order; optimise tree2 against it.
      2. Fix tree2's new leaf order; optimise tree1 against it.
    Stops when the crossing count stops decreasing.

    Parameters
    ----------
    tree1, tree2 : TreeNode
        Roots of the two trees.  Topology is fixed; only child order changes.
    max_iter : int
        Maximum number of full (two-sided) iterations.
    inplace : bool
        If False (default) the trees are deep-copied so the originals are
        unchanged.  Pass True to modify the trees in place.

    Returns
    -------
    order1 : list[str]   – final leaf order for tree1
    order2 : list[str]   – final leaf order for tree2
    crossings : int      – number of crossings in the final layout
    """
    if not inplace:
        tree1 = copy.deepcopy(tree1)
        tree2 = copy.deepcopy(tree2)

    prev_crossings = None

    for iteration in range(max_iter):
        # ── Step 1: fix tree1, optimise tree2 ─────────────────────────────
        order1 = tree1.leaves()
        pos1 = {name: i for i, name in enumerate(order1)}
        optimize_one_side(tree2, pos1)

        # ── Step 2: fix tree2, optimise tree1 ─────────────────────────────
        order2 = tree2.leaves()
        pos2 = {name: i for i, name in enumerate(order2)}
        optimize_one_side(tree1, pos2)

        crossings = count_crossings(tree1.leaves(), tree2.leaves())

        if crossings == prev_crossings:
            break           # converged
        prev_crossings = crossings

    return tree1.leaves(), tree2.leaves(), crossings


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_newick(s: str) -> TreeNode:
    """
    Parse a minimal Newick string into a TreeNode tree.
    Supports names on leaves; branch lengths are ignored.
    Example:  "((A,B),(C,(D,E)))"
    """
    s = s.strip().rstrip(";")

    def _parse(pos):
        node = TreeNode()
        if s[pos] == "(":
            pos += 1          # consume '('
            while True:
                child, pos = _parse(pos)
                node.children.append(child)
                if s[pos] == ",":
                    pos += 1  # consume ','
                elif s[pos] == ")":
                    pos += 1  # consume ')'
                    break
            # optional node name after ')'
            start = pos
            while pos < len(s) and s[pos] not in (",", ")", "(", ":", ";"):
                pos += 1
            node.name = s[start:pos] or None
        else:
            # leaf
            start = pos
            while pos < len(s) and s[pos] not in (",", ")", ":", ";"):
                pos += 1
            node.name = s[start:pos]
        # skip branch lengths
        if pos < len(s) and s[pos] == ":":
            pos += 1
            while pos < len(s) and s[pos] not in (",", ")", ";"):
                pos += 1
        return node, pos

    root, _ = _parse(0)
    return root


# ── Demo ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Two trees with the same 8 leaves but different topologies.
    # Before alignment the orderings produce many crossings.
    nwk1 = "((A,(B,C)),((D,E),(F,(G,H))))"
    nwk2 = "((H,(G,F)),((E,D),(C,(B,A))))"  # essentially reverse topology

    t1 = parse_newick(nwk1)
    t2 = parse_newick(nwk2)

    before = count_crossings(t1.leaves(), t2.leaves())
    print(f"Before alignment:  {t1.leaves()}")
    print(f"                   {t2.leaves()}")
    print(f"  crossings = {before}\n")

    order1, order2, after = align_trees(t1, t2)
    print(f"After alignment:   {order1}")
    print(f"                   {order2}")
    print(f"  crossings = {after}")
