from operator import index
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import ast
import cv2
from scipy.interpolate import interp1d

import random
import math
import networkx as nx
# Criando o grafo

cost = []
img = np.zeros((850,850,3),np.uint8)
imgCopy = np.copy(img)
melhorImg = np.zeros((850,850),np.uint8)
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
                    

# Função para calcular o custo total de uma rota
def calcular_custo_rota(grafo, rota):
    global imgCopy
    custo = 0
    imgCopy = np.copy(img)
    for i in range(len(rota) - 1):
        custo += grafo[rota[i]][rota[i+1]]['weight']
        xy1,yy2 = graph.nodes[rota[i]]['pos'],graph.nodes[rota[i+1]]['pos']
        cv2.line(imgCopy, xy1, yy2, (255,255,255), thickness= 1) 
    xy1,yy2 = graph.nodes[rota[-1]]['pos'],graph.nodes[rota[0]]['pos']
    cv2.line(imgCopy, xy1, yy2, (255,255,255), thickness= 1)
    custo += grafo[rota[-1]][rota[0]]['weight']  # Considerar o retorno à cidade inicial
    return custo

# Perturbação da solução atual: troca duas cidades na rota
def perturbacao(rota):
    nova_rota = rota.copy()
    idx1, idx2 = random.sample(range(len(nova_rota)), 2)
    nova_rota[idx1], nova_rota[idx2] = nova_rota[idx2], nova_rota[idx1]
    return nova_rota

# Algoritmo de Têmpera Simulada para o Problema do Caixeiro Viajante
def tempera_simulada(grafo, temp_inicial, temp_final, taxa_resfriamento):
    global imgCopy,melhorImg
    # Gerar uma solução inicial aleatória
    solucao_atual = list(grafo.nodes)
    random.shuffle(solucao_atual)
    custo_atual = calcular_custo_rota(grafo, solucao_atual)
    temp = temp_inicial
    while temp > temp_final:
        nova_solucao = perturbacao(solucao_atual)
        custo_nova_solucao = calcular_custo_rota(grafo, nova_solucao)
        # Aceitar a nova solução com probabilidade Boltzmann
        if custo_nova_solucao < custo_atual or random.random() < math.exp((custo_atual - custo_nova_solucao) / temp):
            melhorImg = imgCopy
            cost.append(custo_nova_solucao)
            solucao_atual = nova_solucao
            custo_atual = custo_nova_solucao
        # Resfriamento
        temp *= taxa_resfriamento
    return solucao_atual, custo_atual

# Parâmetros iniciais do algoritmo
temp_inicial = 7000
temp_final = 0.1
taxa_resfriamento = 0.98
graph = get_graph()

# Executar a Têmpera Simulada
melhor_rota, melhor_custo = tempera_simulada(graph, temp_inicial, temp_final, taxa_resfriamento)
print("Melhor rota encontrada:", melhor_rota)
print("Custo da melhor rota:", melhor_custo)
plt.ylabel("Custo")
plt.xlabel("Iterações")
plt.plot(cost)
plt.show()
cv2.imshow('Imagem', melhorImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

