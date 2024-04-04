import itertools
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
import random
from scipy.spatial import distance
from random import random, randint, shuffle, sample


def get_graph():
    graph = nx.Graph()
    cvsdata = pd.read_csv('grafo.csv', header=None)
    for value in cvsdata.values:
        cordinate = ast.literal_eval(value[1])
        graph.add_node(value[0],pos= cordinate)
    for u in graph.nodes():
        for v in graph.nodes():
            if u != v:
                pos_u = graph.nodes[u]['pos']
                pos_v = graph.nodes[v]['pos']
                distancia = int(np.linalg.norm(np.array(pos_u) - np.array(pos_v)))
                graph.add_edge(u, v, weight=distancia)
    return graph

global grafo
grafo = get_graph()

class Individuo():
    def __init__(self):
        self.fitness = 0
        self.soma = 0
        self.probabilidade = 0
        self.geracao = 0
        self.graph = grafo
        self.num_cidades = list(self.graph)
        self.cromossomo = []
        self.matriz_adjacencias = nx.to_numpy_array(self.graph)
        cidades = self.num_cidades
        cidade_atual = []
        for i in range(len(self.num_cidades)):
            cidade_atual.append(i)

        while len(cidade_atual) > 0:
            city = randint(0, len(cidade_atual) - 1)
            self.cromossomo.append(cidade_atual.pop(city))



    def print(self):
        print(f'{self.cromossomo},{self.soma},{self.geracao}')

    def calc_fitness(self):
        self.soma = 0

        for i in range(len(self.cromossomo) - 1):
            distancia = self.matriz_adjacencias[self.cromossomo[i]][self.cromossomo[i+1]]
            self.soma += distancia
        distancia = self.matriz_adjacencias[self.cromossomo[i+1]][self.cromossomo[0]]
        self.soma += distancia

    def Reproduzir(self, outro):

        cromossomo_1 = self.cromossomo
        cromossomo_2 = outro.cromossomo

        selecionar_gene1 = randint(5, len(cromossomo_1) - 5)
        selecionar_gene2 = selecionar_gene1

        start_gene=min(selecionar_gene1,selecionar_gene2)
        end_gene=max(selecionar_gene1,selecionar_gene2)

        parte1_cromossomo1 = cromossomo_1[end_gene:]

        parte1_cromossomo2 = cromossomo_2[end_gene:]

        parte2_cromossomo1 =[item for item in cromossomo_2 if item not in parte1_cromossomo1]
        parte2_cromossomo2 =[item for item in cromossomo_1 if item not in parte1_cromossomo2]

        c_filho1 = parte1_cromossomo1 + parte2_cromossomo1
        c_filho2 = parte1_cromossomo2 + parte2_cromossomo2

        filho1 = Individuo()
        filho2 = Individuo()

        filho1.cromossomo = c_filho1
        filho2.cromossomo = c_filho2
        filho1.calc_fitness()
        filho2.calc_fitness()
        filho1.geracao = self.geracao + 1
        filho2.geracao = self.geracao + 1

        return filho1, filho2

    def mutacao(self, taxa_mutacao):
        if randint(1, 100) <= taxa_mutacao:
            genes = self.cromossomo
            gene_1 = randint(0, len(genes) - 1)
            gene_2 = randint(0, len(genes) - 1)
            tmp = genes[gene_1]
            genes[gene_1] = genes[gene_2]
            genes[gene_2] = tmp
        return self

class Populacao():
    def __init__(self):
        self.lista_individuos = []
        pass
    def add_individuo(self, obj_individuo):
        assert isinstance(obj_individuo, Individuo)
        self.lista_individuos.append(obj_individuo)

    def printIndividuos(self):
        for p in self.lista_individuos:
            p.print()
    def mutaIndividuos(self, taxa_mutacao):
        for i in self.lista_individuos:
            i.mutacao(taxa_mutacao)
    def remove_individuo(self, individuo):
        self.lista_individuos.remove(individuo)
    def calcula_fitpop(self):
        for p in self.lista_individuos:
            p.calc_fitness()
    def melhor_indiviuo(self, melhor):
        for i in range(len(self.lista_individuos) - 1):
            if self.lista_individuos[i].soma < melhor.soma:
                melhor = self.lista_individuos[i]
        return melhor
    def sort_population(self):
            self.lista_individuos = sorted(self.lista_individuos,
                                     key=lambda lista_individuos: lista_individuos.soma,
                                     reverse=False)

def Criar_populacao(n_individuos, pop):
    for i in range(n_individuos):
        novo_individuo = Individuo()
        novo_individuo.calc_fitness()
        pop.add_individuo(novo_individuo)

def calcular_probabilidaes(populacao):

    for i in populacao.lista_individuos:
        i.fitness = 1/i.soma
    soma_aptidao = 0
    for i in populacao.lista_individuos:
        soma_aptidao += i.fitness

    for i in populacao.lista_individuos:
        i.probabilidade = i.fitness/soma_aptidao

def selecionar_individuos(populacao):
    pai = -1
    i= 0
    numero_aleatorio = random()
    soma = 0
    while i < len(populacao.lista_individuos) and soma <= numero_aleatorio:
        soma += populacao.lista_individuos[i].probabilidade
        pai += 1
        i += 1
    return pai

def Algoritmogenetico():
    g = get_graph()
    taxa_mutacao = 1
    geracao = 0
    melhor_cromossomo = Individuo()
    melhor_cromossomo.soma = 100000
    pop = Populacao()
    new_pop = Populacao()
    Criar_populacao(10, pop)
    calcular_probabilidaes(pop)
    print("Primeira geração:")
    pop.printIndividuos()
    melhor_cromossomo = pop.melhor_indiviuo(melhor_cromossomo)
    while geracao < 5000:
         calcular_probabilidaes(pop)
         for i in range(5):
             pai1 = pop.lista_individuos[selecionar_individuos(pop)]
             pai2 = pop.lista_individuos[selecionar_individuos(pop)]
             while pai1 == pai2:
                     pai2 = pop.lista_individuos[selecionar_individuos(pop)]
             filho1, filho2 = pai1.Reproduzir(pai2)
             new_pop.add_individuo(filho1)
             new_pop.add_individuo(filho2)
         pop.lista_individuos = list(new_pop.lista_individuos)
         new_pop.lista_individuos.clear()
         pop.mutaIndividuos(taxa_mutacao)
         pop.calcula_fitpop()
         geracao += 1
         melhor_cromossomo = pop.melhor_indiviuo(melhor_cromossomo)
         if melhor_cromossomo not in pop.lista_individuos:
             pop.lista_individuos.append(melhor_cromossomo)
             melhor_cromossomo.geracao  = geracao

    calcular_probabilidaes(pop)
    print("Ultima geração:")
    pop.printIndividuos()
    print("Melhor Individuo")
    melhor_cromossomo.calc_fitness()
    melhor_cromossomo.print()

Algoritmogenetico()
