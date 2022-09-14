import numpy as np
import os
from src.functions import *
import random
from src.predict import predict

class Chromosome:
    def __init__(self, limit=1, size=3, insert=None, delete=None, replace=None):
        self.LIMIT = limit
        self.size = size

        if insert is None:
            insert = random.uniform(0, self.LIMIT)
        if delete is None:
            delete = random.uniform(0, self.LIMIT)
        if replace is None:
            replace = random.uniform(0, self.LIMIT)

        self.genes = np.array([insert, delete, replace])
        self.fitness = .0


class Genetic:
    def __init__(self, dataset='nr', datapath='data', resultpath='result', P_Mutate=0.025, P_Tournament=0.75,
                 R_Mutate=0.15, sizeOfGroup=27, maxGeneration=2000):
        self.P_Mutate = P_Mutate
        self.P_Tournament = P_Tournament
        self.R_Mutate = R_Mutate
        self.SIZE_OF_GROUP = sizeOfGroup
        self.MAX_GENERATION = maxGeneration

        self.method = 'wnngip'
        self.dataset = dataset
        self.chromosomeList = []
        self.datapath = datapath
        self.resultpath = resultpath

        self.path = os.path.join(self.datapath, self.dataset)

        self.fileInfo = []
        for file in os.listdir(self.path):
            with open(os.path.join(self.path, file), 'r') as f:
                self.fileInfo.append((file.split('.')[0], f.read().rstrip()))
        self.fileInfo.sort(key=lambda x: x[0])
        self.fileInfo = np.array(self.fileInfo)

    def initChromosome(self):
        for _ in range(self.SIZE_OF_GROUP):
            self.chromosomeList.append(Chromosome())
            self.evaluate_chromosome(self.chromosomeList[-1])

    def evaluate_chromosome(self, chromosome):
        # Create a Sim_Matrix based on the value of chromosome, and predict using that value
        
        adjMatrix = getAdjMatrix(path=os.path.join(self.datapath, self.dataset), weights=chromosome.genes,
                                 fileInfo=self.fileInfo)

        makeResultFile(adjMatrix, self.dataset, self.fileInfo)

        auc, aupr = predict([f'--method={self.method}',
                             f'--dataset={self.dataset}',
                             f'--specify-arg={1}'  # 0: find optimal setting, 1: default setting
                             ])

        chromosome.fitness = auc
        print(auc)
        if not os.path.isdir(self.resultpath):
            os.mkdir(self.resultpath)
        with open(os.path.join(self.resultpath, f'{self.dataset}.txt'), 'a') as f:
            f.write(f'cost: {np.round_(chromosome.genes, 4)}, fitness: {round(chromosome.fitness, 4)}\n')

    def selectParent(self):
        p1 = random.randrange(self.SIZE_OF_GROUP)
        p2 = random.randrange(self.SIZE_OF_GROUP)
        while p1 == p2:
            p2 = random.randrange(self.SIZE_OF_GROUP)

        if self.chromosomeList[p1].fitness < self.chromosomeList[p2].fitness:
            p1, p2 = p2, p1

        if self.P_Tournament > random.random():
            return p1
        else:
            return p2

    def mutation(self, chromosome):
        for i in range(chromosome.size):
            if random.random() < self.P_Mutate:
                chromosome.genes[i] += random.uniform(-1, 1) * self.R_Mutate
                chromosome.genes[i] = max(chromosome.genes[i], 0)
                chromosome.genes[i] = min(chromosome.genes[i], chromosome.LIMIT)

        return chromosome

    def crossOver(self, p1, p2):
        mother, father = self.chromosomeList[p1], self.chromosomeList[p2]
        offspring = Chromosome()

        offspring.genes = (mother.genes + father.genes) / 2.
        return offspring

    def replace(self, p1, p2, offspring):
        i = p2 if self.chromosomeList[p1].fitness > self.chromosomeList[p2].fitness else p1
        self.chromosomeList[i].genes = offspring.genes
        self.chromosomeList[i].fitness = offspring.fitness

    def print_population_statistics(self, epoch):
        fSum, fMax = .0, -1e9

        for chromosome in self.chromosomeList:
            fSum += chromosome.fitness
            fMax = max(fMax, chromosome.fitness)

        print(
            f'{epoch}/{self.MAX_GENERATION} average fitness: {round(fSum / self.SIZE_OF_GROUP, 4)}, max fitness: {round(fMax, 4)}')

    def getBestFitness(self):
        fMax, iMax = -1e9, -1

        for i, chromosome in enumerate(self.chromosomeList):
            if chromosome.fitness > fMax:
                fMax = chromosome.fitness
                iMax = i
        return iMax

    def run(self):
        self.initChromosome()
        epoch = 0
        while epoch < self.MAX_GENERATION:
            p1 = self.selectParent()
            p2 = self.selectParent()
            if p1 == p2:
                continue

            offspring = self.crossOver(p1, p2)
            self.mutation(offspring)
            self.evaluate_chromosome(offspring)
            self.replace(p1, p2, offspring)
            self.print_population_statistics(epoch)

            epoch += 1

        iMax = self.getBestFitness()
        bestChromosome = self.chromosomeList[iMax]
        print(f'best fitness: {round(bestChromosome.fitness, 4)}, weights: {np.round_(bestChromosome.genes, 4)}')
        return round(bestChromosome.fitness, 4)
