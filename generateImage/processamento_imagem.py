import cv2
import numpy as np
from itertools import combinations

points = []

imagem = cv2.imread('/home/lucas.matheus/Desktop/Screenshot from 2024-03-25 13-13-14.png')


def eliminar_acima_de_limiar(imagem, limiar):
    # Converter a imagem para tons de cinza
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # Aplicar a transformação: Pixels abaixo do limiar permanecem inalterados, outros se tornam brancos (255)
    _, resultado = cv2.threshold(gray, 13, 255, cv2.THRESH_BINARY)

    # # resultado = cv2.medianBlur(resultado, 3)
    # kernel = np.ones((5,5), np.uint8)
    kernel2 = np.ones((3,3), np.uint8)
    
    resultado = cv2.morphologyEx(resultado, cv2.MORPH_CLOSE, kernel2)
    # resultado = cv2.morphologyEx(resultado, cv2.MORPH_CLOSE, kernel2)
    # resultado = cv2.morphologyEx(resultado, cv2.MORPH_CLOSE, kernel)
    # resultado = cv2.dilate(resultado, kernel, iterations=1)
    # contornos, _ = cv2.findContours(resultado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # Encontrar contornos na imagem binária
    # vertices = []
    # for contorno in contornos:
    #     M = cv2.moments(contorno)
    #     if M["m00"] != 0:
    #         cX = int(M["m10"] / M["m00"])
    #         cY = int(M["m01"] / M["m00"])
    #         vertices.append((cX, cY))

    # # Calcular todas as possíveis combinações de pares de vértices para criar as arestas
    # arestas = list(combinations(vertices, 2))

    # # Desenhar as arestas conectando os pares de vértices
    # for aresta in arestas:
    #     cv2.line(resultado, aresta[0], aresta[1], (0, 255, 0), 1)

    return resultado


# Definir o limiar desejado (0-255)
limiar = 20

# Chamar a função para eliminar pixels acima do limiar
imagem_resultante = eliminar_acima_de_limiar(imagem, limiar)
# Exibir a imagem original e a imagem resultante
cv2.imshow('Imagem Original', imagem)
cv2.imshow('Imagem Resultante', imagem_resultante)

cv2.waitKey(0)
cv2.destroyAllWindows()