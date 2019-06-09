import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 4            # DNA length
POP_SIZE = 4           # population size
CROSS_RATE = 0.8         # mating probability (DNA crossover)
MUTATION_RATE = 0.003    # mutation probability
N_GENERATIONS = 1
X_BOUND = [0.5, 2.5]         # x upper and lower bounds


def f(x): return (np.exp(x) * np.sin(10*np.pi*x) + 1) / x  # to find the maximum of this function


# find non-zero fitness for selection
def get_fitness(pred):
    return pred + 0.001 - np.min(pred)


# convert binary DNA to decimal and normalize it to a range(0.5, 2.5)
def normalize_to_range(population):
    return bin2dec(population) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1])


def bin2dec(pop):
    # return pop.dot(2 ** np.arange(DNA_SIZE)[::-1])

    output = []
    for dna in pop:
        s = [str(i) for i in dna]
        res = int("".join(s), 2)
        output.append(res)
    return np.array(output)


# wheel roulette selection
def select(pop, fitness):
    idx = np.random.choice(np.arange(POP_SIZE),
                           size=POP_SIZE,
                           replace=True,
                           p=fitness/fitness.sum())
    return pop[idx]


def crossover(parent, pop):     # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE:
        idx_ = np.random.randint(0, POP_SIZE, size=1)           # select another individual from pop
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)   # choose crossover points
        # print("cross_point", cross_points)
        # print("parent[cross_point", parent[cross_points])
        parent[cross_points] = pop[idx_, cross_points]          # mating and produce one child

    return parent


def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))   # initialize the pop DNA


x = np.linspace(0.5, 2.5, 200)
# print(f(x))
max_y = max(f(x))
max_x = x[f(x).argmax()]
print (max_x, max_y)
plt.plot(x, f(x))
plt.plot(max_x, max_y, '*')

for _ in range(N_GENERATIONS):

    f_values = f(normalize_to_range(pop))    # compute function value by extracting DNA
    print("F_Values", f_values)

    if 'sca' in globals(): sca.remove()
    sca = plt.scatter(normalize_to_range(pop), f_values, s=200,  lw=0, c='red', alpha=0.5)
    plt.pause(0.05)

    # GA part (evolution)
    fitness = get_fitness(f_values)
    print("get_fitness", fitness)
    # print("Most fitted DNA: ", pop[np.argmax(fitness), :])
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child       # parent is replaced by its child
plt.show()