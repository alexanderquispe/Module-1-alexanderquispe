from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    
    # Convert the vals tuple to a list to modify
    vals_plus_epsilon = list(vals)
    vals_minus_epsilon = list(vals)
    
    # Increment and decrement the value at index `arg`
    vals_plus_epsilon[arg] += epsilon
    vals_minus_epsilon[arg] -= epsilon
    
    # Compute the central difference approximation
    f_plus = f(*vals_plus_epsilon)
    f_minus = f(*vals_minus_epsilon)
    
    return (f_plus - f_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    order: List[Variable] = []
    seen = set()

    def visit(var: Variable) -> None:
        if var.unique_id in seen or var.is_constant():
            return
        if not var.is_leaf():
            for m in var.parents:
                if not m.is_constant():
                    visit(m)
        seen.add(var.unique_id)
        order.append(var)  # Use append here

    visit(variable)
    return reversed(order)  # Reverse the order at the end



def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    
    # # First, get the topological order of variables

    # topological_order = topological_sort(variable)
    
    # # Initialize a dictionary to store the derivatives of each variable
    # derivatives = {variable.unique_id: deriv}
    
    # # Iterate over the variables in reverse topological order
    # for var in topological_order:
    #     # Get the derivative for the current variable
    #     d_output = derivatives.get(var, 0)
        
    #     # If the variable is a leaf, accumulate the derivative
    #     if var.is_leaf():
    #         var.accumulate_derivative(d_output)
    #     else:
    #         # Otherwise, propagate the derivative to the parents using the chain rule
    #         for parent, local_derivative in var.chain_rule(d_output):
    #             # Accumulate the derivative for each parent
    #             if parent in derivatives:
    #                 derivatives[parent] += local_derivative
    #             else:
    #                 derivatives[parent] = local_derivative

    # Step 1: Get the topological order of the computation graph
    queue = topological_sort(variable)

    # Step 2: Initialize a dictionary to store derivatives
    derivatives = {}

    # Step 3: Set the derivative for the output variable
    derivatives[variable.unique_id] = deriv

    # Step 4: Iterate over the variables in the queue (in reverse topological order)
    for var in queue:
        # Get the current derivative of this variable
        deriv = derivatives[var.unique_id]

        # If it's a leaf node, accumulate its derivative
        if var.is_leaf():
            var.accumulate_derivative(deriv)

        # Otherwise, propagate the derivative backward using the chain rule
        else:
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue

                # Step 5: Use setdefault to initialize the derivative if it doesn't exist
                derivatives.setdefault(v.unique_id, 0.0)

                # Step 6: Add the current contribution to the derivative
                derivatives[v.unique_id] += d

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
