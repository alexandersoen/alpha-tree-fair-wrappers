
import graphviz
from queue import Queue
from typing import TypeVar, Generic, Callable, Optional, List
from enum import Enum, auto
from functools import wraps
from dataclasses import dataclass

T = TypeVar('T')
L = TypeVar('L')

TDirection = TypeVar('TDirection', bound='Direction')

H = Callable[[T], TDirection]

class Direction(Enum):
    LEFT = auto()  # LEFT is true on the rule
    RIGHT = auto()

# Node selection data class
@dataclass
class RuleSelection(Generic[T]):
    hypothsis: H
    local_ent_drop: float
    split_index: Optional[int] = None

    def __call__(self, x: T) -> TDirection:
        return self.hypothsis(x)

## Decorator functions for fixing outputs
def float_rule_wrapper(func: Callable[[T], float], thresh: float = 0) -> H:
    @wraps(func)
    def wrapped(x: T) -> TDirection:
        return Direction.LEFT if func(x) > thresh else Direction.RIGHT
    return wrapped

def bool_rule_wrapper(func: Callable[[T], bool]) -> H:
    @wraps(func)
    def wrapped(x: T) -> TDirection:
        return Direction.LEFT if func(x) else Direction.RIGHT
    return wrapped
## Decorator functions for fixing outputs

@dataclass
class DecisionNode(Generic[T, L]):
    value: Optional[L]
    left: Optional['DecisionNode[T, L]'] = None
    right: Optional['DecisionNode[T, L]'] = None
    rule: Optional[RuleSelection] = None
    is_leaf: bool = True
    parent: 'DecisionNode[T, L]' = None

    def split_node(self, rule: RuleSelection, left_val: L, right_val: L):
        self.value = None  # don't really need to do this
        self.is_leaf = False
        self.rule = rule

        self.left = DecisionNode[T, L](value = left_val, parent = self)
        self.right = DecisionNode[T, L](value = right_val, parent = self)

    def find_leaf(self, x: T) -> 'DecisionNode[T, L]':
        if self.is_leaf:
            return self

        dir = self.rule(x)
        if dir == Direction.LEFT:
            return self.left.find_leaf(x)
        else:
            return self.right.find_leaf(x)

    def __call__(self, x: T) -> L:
        return self.find_leaf(x).value

    def __hash__(self) -> int:
        return hash(repr(self))

class DecisionTree(Generic[T, L]):
    root: DecisionNode[T, L]

    def __init__(self, init_label: L) -> None:
        self.root = DecisionNode[T, L](value = init_label)

    def find_leaf(self, x: T) -> DecisionNode[T, L]:
        return self.root.find_leaf(x)

    @property
    def leaves(self) -> List[DecisionNode[T, L]]:
        llist = []
        nlist = [self.root]
        while len(nlist) > 0:
            cur_dn = nlist.pop(0)
            if cur_dn.is_leaf:
                llist.append(cur_dn)
            else:
                nlist = [cur_dn.left, cur_dn.right] + nlist
        return llist

    def __call__(self, x: T) -> L:
        return self.root(x)

    def __hash__(self) -> int:
        return hash(repr(self))

    def basic_tree_dot(self, label_names):
        dot = graphviz.Digraph(comment='')
        counter = 0
        node_queue = Queue()
        node_queue.put((self.root, None))
        while not node_queue.empty():
            cur_node, parent = node_queue.get()
            if cur_node.is_leaf:
                cur_label = '{:.2f}'.format(cur_node.value)
                dot.node(str(counter), cur_label, shape='circle')
            else:
                # Generate labels
                if cur_node.rule.split_index is None:
                    cur_label = 'age'
                else:
                    cur_label = label_names[cur_node.rule.split_index].split('__')[1]
                
                # Update dot
                dot.node(str(counter), cur_label)

            # Update edge
            if parent is not None:
                dot.edge(str(parent), str(counter))

            # Add children
            if cur_node.left:
                node_queue.put((cur_node.left, counter))
            if cur_node.right:
                node_queue.put((cur_node.right, counter))

            counter += 1

        return dot