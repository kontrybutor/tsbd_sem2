import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 20  # DNA length
POP_SIZE = 1000  # population size
CROSS_RATE = 0.4  # mating probability (DNA crossover)
MUTATION_RATE = 0.001  # mutation probability
N_GENERATIONS = 100  # number of generations
X_MIN = 0.5  # x min bound
X_MAX = 2.5  # x upper bound


class GA:
    def __init__(self, dna_size, pop_size, cross_rate, mutation_rate,
                 no_of_generations, x_min, x_max):
        self.dna_size = dna_size
        self.pop_size = pop_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.no_of_generations = no_of_generations
        self.x_min = x_min
        self.x_max = x_max
        self.pop = np.random.randint(2, size=(self.pop_size, self.dna_size))  # initialize the pop DNA
    
    @staticmethod
    def f(x):
        return (np.exp(x) * np.sin(10 * np.pi * x) + 1) / x
    
    # find non-zero fitness for selection
    @staticmethod
    def get_fitness(values):
        return values + 0.001 - np.min(values)
    
    # convert binary DNA to decimal and normalize it to a range(0.5, 2.5)
    def normalize_to_range(self, population):
        x = self.bin2dec(population)
        # print(x)
        numerator = (self.x_max - self.x_min) * (x - np.min(x))
        denominator = np.max(x) - np.min(x)
        normalized = (numerator / denominator) + self.x_min
        
        return normalized
    
    @staticmethod
    def bin2dec(pop):
        output = []
        for dna in pop:
            s = [str(i) for i in dna]
            res = int("".join(s), 2)
            output.append(res)
        return np.array(output)
    
    # wheel roulette selection
    def select(self, pop, fitness):
        idx = np.random.choice(np.arange(self.pop_size),
                               size=self.pop_size,
                               replace=True,
                               p=fitness / fitness.sum())
        return pop[idx]
    
    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            idx_ = np.random.randint(0, self.pop_size, size=1)
            cross_points = np.random.randint(0, self.dna_size, size=np.random.randint(1, 2*self.dna_size))  # choose crossover points
            parent[cross_points] = pop[idx_, cross_points]  # mating and produce one child
        
        return parent
    
    @staticmethod
    def mutate(child):
        for point in range(DNA_SIZE):
            if np.random.rand() < MUTATION_RATE:
                child[point] = 1 if child[point] == 0 else 0
        return child
    
    def evolve(self):
        x = np.linspace(0.5, 2.5, 10000)
        max_y = max(self.f(x))
        max_x = x[self.f(x).argmax()]
        plt.plot(x, self.f(x))
        plt.plot(max_x, max_y, '*')
        print("max_y", max_y, "max_x", max_x)
        
        max_found_y = 0.0
        max_found_x = 0
        iteration = 0
        f_mean = []
        
        for i in range(self.no_of_generations):
            normalized = self.normalize_to_range(self.pop)
            f_values = self.f(normalized)
            
            plt.scatter(normalized, f_values, s=40, lw=0, c='red', alpha=0.5);
            plt.pause(0.05)
            
            # evolution...
            fitness = self.get_fitness(f_values)
            # print("Fitness", fitness)
            f_mean.append(np.mean(fitness))
            if max(f_values) > max_found_y:
                max_found_y = max(f_values)
                max_found_x = normalized[f_values.argmax()]
                iteration = i
            probably_best_parent = self.select(self.pop, fitness)
            probably_best_parent_copy = probably_best_parent.copy()
            for parent in self.pop:
                child = self.crossover(parent, probably_best_parent_copy)
                child = self.mutate(child)
                parent[:] = child  # parent is replaced by its child
        plt.show()
        print("y =", round(max_found_y, 3), " in", iteration, "iteration")
        print("x =", round(max_found_x, 3), " in", iteration, "iteration")
        plt.plot(x, self.f(x))
        plt.plot(max_found_x, max_found_y, '*', c='blue')
        plt.show()
        plt.plot(range(self.no_of_generations), f_mean)
        plt.xlabel("generacja")
        plt.ylabel("wartość średnia funkcji dopasowania")
        plt.show()


def main():
    ga = GA(DNA_SIZE, POP_SIZE, CROSS_RATE, MUTATION_RATE, N_GENERATIONS, X_MIN, X_MAX)
    ga.evolve()


if __name__ == "__main__":
    main()
