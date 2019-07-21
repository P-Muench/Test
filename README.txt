This package allows solving the ABMCP and RABMCP via the algorithms of the thesis.
A benchmarking script for each of the problems is present that generates, based on given parameters, instances at random and tests all algorithms on them. These 2 scripts have been used as such to generate the test instances used in the thesis.
The Fix-and-Spread algorithm is accessible via the Optimizer class that is easily configurable. Examples of that can be found in the Benchmarking_ABMCP.py script.
Examples:
	# opt is an Optimizer instance
	opt.set_solution_method(Optimizer.COLUMN_GENERATION)	# Different solution methods available, see constants in class Optimizer
	opt.high_multiplicty = True		# set if high multiplicity algorithms shall be used
	opt.set_timelimit_subproblem(t_limit)	# impose a timelimit to the subproblem
	opt.JSA_method = Optimizer.JSA_APPROX	# set the method to solve the Job Shop Scheduling problem
	opt.choose_paths_randomly = True		# Select if a random ordering of operations shall be used before scheduling
	opt.opt_choose_paths = True				# If True, after an initial path for every job has been chosen, use an IP for minimize parallel operations on a machine
											# Takes some time but might improve performance
	opt.JSA_repetitions = reps				# Set the number of repetitions for the (randomized) Job Shop Algorithm



Folders:
- Datastructures: Classes for things like Machines, Abilities and Graphs 
- Heuristics: All the algorithms. Provides solve_low_multiplicity and solve_high_multiplicty functions each.
- RandomInputGeneration: Includes methods to generate (and write) random instances.