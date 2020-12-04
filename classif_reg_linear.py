# -*- coding: utf-8 -*-
"""
Aluno: Cleiton Moya de Almeida
CPS844 - Inteligência Computacional I
Classificação por Regressão Linear
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import linalg
import random

# Seed para fins de depuração
#seed = random.randrange(2**32)
seed = 3727339038
np.random.seed(seed)
random.seed(seed)

""" ******************************** NOTAÇÃO ******************************** 

------- VETORES -------
N:  Número de pontos da amostra
Xe: Espaço de entrada x1 x x2 = [x11, x12, x21, x22]
X1: Vetor (Nx1) com as coordenadas 'x1' dos pontos de amostra
X2: Vetor (Nx1) com as coordenadas 'x2' dos pontos de amostra
X:  Matriz (Nx3) dos pontos de amostra aumentada com X0 = 1: [[1], [X1], [X2]]
Y:  Vetor (Nx1) com as coordenadas 'y' dos pontos de amostra
H:  Vetor (Nx1) com as hipóteses calculadas para o vetor X
G:  Vetor (Nx1) com as hipóteses finais calculadas para o vetor X 

------- FUNÇÕES -------
f:   função alvo
f_:  reta da função-alvo
h:   função hipótese
h_:  reta da função hipótese
g:   hipótese final selecionada
"""

# **************************** FUNÇÕES AUXILIARES **************************** 

# A.1 Gera N pontos aleatórios do espaço de entrada X
def randomXPoints(N,Xe):
    X1 = np.random.uniform(Xe[0], Xe[1], [N,1])
    X2 = np.random.uniform(Xe[2], Xe[3], [N,1])
    return X1,X2

# A.2 Gera uma função-alvo aleatória no espaço de entrada 'X'
#    Função alvo:          f = sign(x2 - f(x1))
#    Reta da função alvo:  f_ = a*x1 + b
#    retorna f_, f      
def randomf(Xe):
    
    # escolhe 2 pontos aleatórios (x1, x2) no espaço de entrada
    X1,X2 = randomXPoints(2,Xe)
    x11 = X1[0,0]
    x12 = X1[1,0]
    x21 = X2[0,0]
    x22 = X2[1,0]

    # avalia se (x1p1 != x1p2)
    while x11 == x21:
        # caso sim, escolhe outro ponto p2
         p2 = randomXPoints(1,Xe)
         x21 = p2[0,0]
         x22 = p2[1,0]
    
    # calcula as parâmetros 'a' e 'b' da reta 'x2 = a*x1 + b'
    a = (x22-x12)/(x21-x11)
    b = x12-a*x11
    
    # função da reta da função-alvo
    def f_(x1):
        return a*x1 + b
    
    # função-alvo
    def f(x1,x2):
        return np.sign(x2-f_(x1))
    
    return f_, f

# A.3 Função hipótese para um dado vetor de pesos 'w' 
def h(w):
    # Para x,w vetores-colunas, a fómula é h = sign(w.T@x), h escalar
    # Se X é uma matriz na forma padronizada pelo alg. de regressão,
    # a fórmula se torna h = X@w, h,w vetores-colunas.
    h = lambda x: np.sign(x@w)
    return h

# A.4 Reta da função-hipótese para um dado vetor de pesos 'w'
def h_(w):
    a = -w[1,0]/w[2,0]
    b = -w[0,0]/w[2,0]
    
    def h1_(x1):
        return a*x1 + b
    
    return h1_

# A.5 Seleciona de forma aleatória um ponto classificado incorretamente
#   Caso todos os pontos estiverem classifcados corretamente, retorna '-1'     
def randonMalClassificado(Y,H):
    C = (H != Y) # matriz de comparação de 'H' com 'Y'
    
    # 'M': lista com as posições em '(X,Y)' dos elementos mal-classificados
    M = np.array([]) 
    for idx, val in enumerate(C):
        if (val == True):
            # vetor com as posições dos elementos mal-classificados
            M = np.append(M,idx) 
    
    # se núm elem. de M é > 0, escolhe um elemento aleatório 
    if (np.size(M)>0):
        return int(np.random.choice(M))
    else:
        return -1

# A.6 Estimativa de Eout para N pontos de fora da amostra
def avaliaEoutExp(f,g,N):
    # gera aleatoriamente N pontos em X1, X2
    X1out, X2out = randomXPoints(N,Xe)
    Xout = np.hstack((np.array([np.ones(N)]).T,X1out,X2out)) # inserção x0 = 1 

    # calcula Eout
    Yout = f(X1out,X2out)
    Eout = np.mean(g(Xout)!=Yout)
    return Xout, Yout, Eout

# A.7 Estima w usando regressão linear
def regressaoLinear(X,Y):
    # 7.1 Computa a pseudo-inversa de X'
    pinvX = linalg.pinv(X)    
    # 7.2 Cálcula a hipótese wlin
    wlin = pinvX@Y    
    return wlin

# A.8 Introduz ruído em um conjunto de pontos (X, Y)
# p: percentual de pontos que será introduzido ruído
def introduzRuido(X1,X2,Y,p):
    N = np.size(Y)
    rndPos = random.sample(range(N), int(p*N))
    for j in rndPos:
        Y[j] = -np.sign(Y[j]) # inverte o valor
    return Y

    N = np.size(Y)
    rndPos = random.sample(range(N), int(p*N))
    for j in rndPos:
        Y[j] = -np.sign(Y[j]) # inverte o valor
    return Y


# *********************** PARÂMETROS DE SIMULAÇÃO ***************************

Xe = np.array([-1, 1, -1, 1])   # Espaço de entrada
nexp = 1000  # número de experimentos
N =    100   # número de pontos do conjunto de dados
Nout = 1000  # numero de pontos fora da amostra p/ cálculo de Eout
pr =   0     # probabilidade de ruído nos dados de treinamento

# Vetores auxiliares
T = np.array([])    # número de iterações que cada experimento convergiu
Ein = np.array([])  # vetor com Ein estimado de cada experimento
Eout = np.array([]) # vetor com Eout estimado de cada experimento


# **********************  EXECUÇÃO DOS EXPERIMENTOS  ***********************
for n in range(nexp):
    
    # 1. Gera uma função alvo-aleatória 'f' e sua função de reta 'f_'
    f_,f = randomf(Xe)
    
    # 2. Gera N exemplares aleatórios (X, Y) / Y!=0
    X1,X2 = randomXPoints(N,Xe)  # escolhe N pontos aleat. no espaço X
    Y = f(X1,X2) 
    while 0 in Y: # Caso tenha algum y=0, substitui por um novo X2 aleat.
        print("Experimento",n)
        pos = np.where(Y==0)
        print("Y tem 0 nas posições ",pos)
        for p in pos[0]:
            _, X2[p]= randomXPoints(1)
        Y = f(X1,X2)

    # 3. Insere a coordenada x0 = 1 no vetor X
    X = np.hstack((np.array([np.ones(N)]).T,X1,X2))

    # 4. Introduz ruído nos dados de treinamento
    if pr> 0: Y = introduzRuido(X1, X2, Y, pr)

    # 5. Executa o algoritmo de Regressão Linear para estimar w
    w = regressaoLinear(X, Y)
    
    # 6. Usa calcula a função hipótese
    g = h(w)
    g_ = h_(w)
    G = g(X)
    
    # 7. Avalia Ein do experimento:
    Ein_n = np.mean(G!=Y)
    Ein = np.append(Ein, Ein_n)
    
    # 8. Avalia Eout do experimento:
    if Nout > 0:
        Xout, Yout, Eout_n = avaliaEoutExp(f,g,Nout)
        Eout = np.append(Eout, Eout_n)
    

# ************************** IMPRIME OS RESULTADOS **************************

if np.size(T)>0: # Caso houver convergência
    print("Convergência:")
    print("\t Médio:", T.mean())
    print("\t Mín.:", T.min())
    print("\t Máx.:", T.max()) 

print("\nEin")
print("\t Médio:", Ein.mean())
print("\t Mín.:", Ein.min())
print("\t Máx.:", Ein.max()) 

if Nout > 0:    
    print("\nEout:")
    print("\t Médio:", Eout.mean())
    print("\t Mín.:", Eout.min())
    print("\t Máx.:", Eout.max()) 


# ***************************** GERA OS GRÁFICOS *****************************

# 0. Configurações do gráfico
plt.figure(1,figsize=(9,9))
plt.xticks(np.linspace(-1, +1, 21))
plt.yticks(np.linspace(-1, +1, 21))
plt.xlim(-1.1,1.1)
plt.ylim(-1.1,1.1)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.grid(linestyle="--")
eX1 = np.linspace(-1, +1, 201) # eixo x1, epaçamento = 0.01

# 1. Plota reta da função-alvo
plt.plot(eX1,f_(eX1))

# 2. Plota a reta da hipótese final 'g'
if w[2] != 0: # necessário para o cálculo dos parâmetros 'a' e 'b' da reta
    plt.plot(eX1,g_(eX1))
else:
    # Caso w[2]=0
    print("ATENÇÃO: w[2]=0!")
    plt.axvline(0, -1, 1, c='m')

# 3. Separa e plota os dados (xn, yn)
X1a = np.array([])
X1v = np.array([])
X2a = np.array([])
X2v = np.array([])
for idx,val in enumerate(Y):
    if val==1:     
        X1a = np.append(X1a,X1[idx])
        X2a = np.append(X2a,X2[idx])
    else:
        X1v = np.append(X1v,X1[idx])
        X2v = np.append(X2v,X2[idx])

colors = ['blue','red']
cmap = matplotlib.colors.ListedColormap(colors)
plt.scatter(X1a ,X2a,c='blue',cmap=cmap)
plt.scatter(X1v ,X2v,c='red',cmap=cmap)
    
# 4. Plota os pontos utilizados para o cálculo de Eout
if Nout > 0:    
    X1o = Xout[:,1]
    X2o = Xout[:,2]
    X1oa = np.array([])
    X1ov = np.array([])
    X2oa = np.array([])
    X2ov = np.array([])
    for idx,val in enumerate(Yout):
        if val==1:     
            X1oa = np.append(X1oa,X1o[idx])
            X2oa = np.append(X2oa,X2o[idx])
        else:
            X1ov = np.append(X1ov,X1o[idx])
            X2ov = np.append(X2ov,X2o[idx])
    
    plt.scatter(X1oa,X2oa,marker='.',alpha=0.5,linewidths=0,c='blue',cmap=cmap)
    plt.scatter(X1ov,X2ov,marker='.',alpha=0.5,linewidths=0,c='red',cmap=cmap)
    plt.legend(['$f$','$g$','$y_n = +1$','$y_n = -1$',
                '$y_{out} = +1$','$y_{out} = -1$'])
else:
    plt.legend(['$f$','$g$','$y_n = +1$','$y_n = -1$'])