import numpy as np
from digital_twin import *
from src.model.dt_model import *
from pyrol import getCout, Objective, Problem, Solver, getParametersFromXmlFile, Bounds
from pyrol.vectors import NumPyVector as myVector
import copy
import scipy.optimize as opt
from scipy.linalg import cho_solve, cho_factor
from scipy.optimize import minimize

def system_identification(dt, input_vars={}, pyrol_file=None):
    if pyrol_file is not None:

        class opt_objective(Objective):
            def __init__(self):
                super().__init__()

            def value(self, control, tol):
                dt.model.update_strength_factor(control)
                obj = copy.copy(dt.model.objective())
                return obj

            def gradient(self, grad, control, tol):
                dt.model.update_strength_factor(control)
                g = dt.model.gradient()
                for ielem in range(dt.model.nelem):
                    grad[ielem] = copy.copy(g[ielem])
                dt.model.add_results_to_movie()

        # Read inputs
        lower_limit = input_vars.get('lower_limit', 1.0e-01)
        upper_limit = input_vars.get('upper_limit', 1.0)

        # Configure parameter list.
        params = getParametersFromXmlFile(pyrol_file)

        # Set the output stream.
        stream = getCout()

        # Set bounds.
        lower_bounds = myVector(np.full(dt.model.nelem, lower_limit))
        upper_bounds = myVector(np.full(dt.model.nelem, upper_limit))
        bounds = Bounds(lower_bounds, upper_bounds)

        # Control and gradient variables
        control = myVector(np.ones(dt.model.nelem, dtype=np.float64))
        for icontrol in range(dt.model.nelem):
            control[icontrol] = dt.model.strength_factor[icontrol]
        grad = control.dual()

        # Set up the problem.
        objective = opt_objective()
        problem = Problem(objective, control, grad)
        problem.addBoundConstraint(bounds)

        # Solve.
        solver = Solver(problem, params)
        solver.solve(stream)

    else:

        def value(control):
            dt.model.update_strength_factor(control)
            obj = copy.copy(dt.model.objective())
            return obj

        def gradient(control):
            dt.model.update_strength_factor(control)
            grad = dt.model.gradient()
            dt.model.add_results_to_movie()
            return grad

        # Read inputs
        lower_limit = input_vars.get('lower_limit', 1.0e-01)
        upper_limit = input_vars.get('upper_limit', 1.0)

        # Bounds
        lower_bounds = np.full(dt.model.nelem, lower_limit)
        upper_bounds = np.full(dt.model.nelem, upper_limit)
        bounds = opt.Bounds(lower_bounds, upper_bounds)

        # Control and gradient variables
        control = np.ones(dt.model.nelem, dtype=np.float64)
        for icontrol in range(dt.model.nelem):
            control[icontrol] = dt.model.strength_factor[icontrol]
        grad = np.zeros(dt.model.nelem, dtype=np.float64)

        res = minimize(value, control, method='L-BFGS-B', jac=gradient, bounds=bounds, \
                       options={'gtol': 1e-6, 'disp': True})
