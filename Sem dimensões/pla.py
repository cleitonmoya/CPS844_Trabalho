# -*- coding: utf-8 -*-
"""
Aluno: Cleiton Moya de Açmeida
CPS844 - Inteligência Computacional I
Perceptron Learning Algorithm (com inicialização por regressão linear)
"""
import numpy as np
from scipy import linalg
import matplotlib
import matplotlib.pyplot as plt
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
X1: Vetor (1xN) com as coordenadas 'x1' dos pontos de amostra
X2: Vetor (1xN) com as coordenadas 'x2' dos pontos de amostra
X:  Matriz (3xN) dos pontos de amostra aumentada com X0 = 1: [[1], [X1], [X2]]
Y:  Vetor (1xN) com as coordenadas 'y' dos pontos de amostra
H:  Vetor (1xN) com as hipóteses calculadas para o vetor X
G:  Vetor (1xN) com as hipóteses finais calculadas para o vetor X 

------- FUNÇÕES -------
f:  função alvo
f_: reta da função-alvo
h:  função hipótese
hm: função da hipótese na iteração 't'
h_: reta da função hipótese
g:  hipótese final selecionada
"""

# **************************** FUNÇÕES AUXILIARES ****************************

# A.1 Gera N pontos aleatórios do espaço de entrada X
def randomXPoints(N,Xe):
    X1 = np.random.uniform(Xe[0], Xe[1], N)
    X2 = np.random.uniform(Xe[2], Xe[3], N)
    return X1,X2

# A.2 Gera uma função-alvo aleatória no espaço de entrada 'X'
#    Função alvo:          f = sign(x2 - f(x1))
#    Reta da função alvo:  f_ = a*x1 + b
#    retorna f_, f      
def randomf(Xe):
    
    # escolhe 2 pontos aleatórios (x1, x2) no espaço de entrada
    X1,X2 = randomXPoints(2,Xe)
    x11 = X1[0]
    x12 = X1[1]
    x21 = X2[0]
    x22 = X2[1]

    # avalia se (x1p1 != x1p2)
    while x11 == x21:
        # caso sim, escolhe outro ponto p2
         p2 = randomXPoints(1,Xe)
         x21 = p2[0]
         x22 = p2[1]
    
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
    h = lambda x: np.sign(w@x)
    return h

# A.4 Reta da função-hipótese para um dado vetor de pesos 'w'
def h_(w):
    a = -w[1]/w[2]
    b = -w[0]/w[2]
    
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
    Xout = np.vstack((np.ones(N),X1out,X2out)) # inserção da coordenada x0 = 1 

    # calcula Eout
    Eout = np.mean(g(Xout)!=f(X1out,X2out))
    return Xout, Eout

# A.7 Estima w usando regressão linear
def regressaoLinear(X,Y):
    # 7.1 Computa a pseudo-inversa de X'
    pinvX = linalg.pinv(X)    
    # 7.2 Cálcula a hipótese wlin
    wlin = pinvX@Y    
    return wlin


# *********************** PARÂMETROS DE SIMULAÇÃO ***************************

Xe = np.array([-1, 1, -1, 1])   # Espaço de entrada
nexp = 1     # número de experimentos     
nit =  5000     # número de iterações máximas de cada experimento
N =    10       # número de pontos do conjunto de dados
Nout = 1000     # numero de pontos fora da amostra p/ cálculo de Eout
Reg = False      # inicializa w0 com regressão linear


# **********************  EXECUÇÃO DOS EXPERIMENTOS  ***********************

# Vetores auxiliares
T = np.array([])        # número de iterações que cada experimento convergiu
Ein = np.array([])      # vetor com Ein estimado de cada experimento
Eout = np.array([])     # vetor com Eout estimado de cada experimento

# Para cada experimento: 
for n in range(nexp):
        
    # 1. Gera uma função alvo-aleatória 'f' e sua função de reta 'f_'
    f_,f = randomf(Xe)
    
    # 2. Gera N exemplares aleatórios (X, Y) / Y!=0
    X1,X2 = randomXPoints(N,Xe)  # escolhe N pontos aleat. no espaço X
    Y = f(X1,X2)
    while 0 in Y: # Caso tenha algum y=0
        print("Experimento",n)
        pos = np.where(Y==0)
        print("Y tem 0 nas posições ",pos)
        for p in pos[0]:
            _, X2[p]= randomXPoints(1)
        Y = f(X1,X2)
    X = np.vstack((np.ones(N),X1,X2)) # inserção da coordenada x0 = 1
    

    # 3. Executa o PLA
    # -------------------- Perceptron Learning Algorithm ---------------------
    # 3.0 Hipótese inicial
    if Reg:
        w = regressaoLinear(X.T,Y) # inicializa 'w' por regressão linear
    else:
        w = np.array([0,0,0])    
    hm = h(w) # função da hipótese inicial
    H = hm(X) # classificação inicial de 'X'
    
    for t in range(nit):
           
        # 3.1 escolhe um ponto mal-classificado, se existir
        pos_x = randonMalClassificado(Y, H)         
        if pos_x == -1: # se todos os pontos estão corretamente classificados
            T = np.append(T,t) # armazena em qual t o algoritmo convergiu
            break          
        if (t==nit-1):
            print("ATENÇÃO: Algoritmo não convergiu para o exp. ", n)           
        x = X[:,pos_x]
        y = Y[pos_x]
        
        # 3.2 Atualiza o vetor de pesos
        w = w + y*x
        
        # 3.3 Gera a hipótese 'hm' com base no vetor 'w' e classifica 'X'
        hm = h(w) # hipótese m
        H = hm(X) # classificação de 'X'
    
    # 3.4 hipótese final selecionada
    g = hm
    G = g(X)
    #  ----------------------------- fim do PLA  -----------------------------
    
    # 4. Avalia o Ein do experimento:
    Ein_n = np.mean(G!=Y)
    Ein = np.append(Ein, Ein_n)
    
    # 5. Avalia o Eout do experimento:
    Xout, Eout_n = avaliaEoutExp(f,g,Nout)
    X1o = Xout[1]
    X2o = Xout[2]
    Yout = f(X1o,X2o)
    Eout = np.append(Eout, Eout_n)
    
    # ------------------------ fim do experimento n -------------------------- 
# ------------------------- fim dos nit experimentos -------------------------  


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

# 1. Plota a reta da função-alvo 'f'
plt.plot(eX1,f_(eX1))

# 2. Plota a reta da hipótese final 'g'
g_ = h_(w)
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
    
# 4. Plota os dados utilizados para o cálculo de Eout
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