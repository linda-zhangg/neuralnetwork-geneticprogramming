# template for initialisers retrieved from: 
# https://github.com/DEAP/deap/blob/cb20d979d3b62635cc330a9804aeb29523bffd42/examples/gp/symbreg.py

import operator
import math
import random

from deap import algorithms, base, creator, tools, gp

# Define the input and output data
data = []
with open('regression.txt') as f:
    next(f)
    next(f)
    for line in f:
        x, y = line.split()
        data.append((float(x), float(y)))

# Define the function set
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1
pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)

# Define the terminal set
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset.renameArguments(ARG0='x')

# Define the fitness and Individual objects
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# Define parameters for the generated individuals
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Define the fitness function
def fitnessFunction(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the absolute error between the expression and the output
    abserrors = (abs(func(x) - y) for x, y in points)
    return math.fsum(abserrors) / len(points),

# Define the genetic operators (evaluation, selection, crossover/mating, mutation)
toolbox.register("evaluate", fitnessFunction, points=data)
toolbox.register("select", tools.selTournament, tournsize=3) # tournament size = 3
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Define the limitations of the individuals (max depth = 17)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    random.seed(135) # change for a different GP run

    pop = toolbox.population(n=500) # population size = 500
    hof = tools.HallOfFame(1) # select the best solution from each generation

    generation = 0
    minError = 100
    while generation < 150 and minError > 0.2: # stop after 150 generations or when the error is less than 0.2
        pop,log = algorithms.eaSimple(pop, toolbox, 0.8, 0.2, 1, stats=None,
                                   halloffame=hof, verbose=False)
        minError = fitnessFunction(hof[0], data)[0]
        generation += 1
        print("Generation: ", generation, "Error: ", minError)
    
    print("Best individual is: ", hof[0])


if __name__ == "__main__":
    main()