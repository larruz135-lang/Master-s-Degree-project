import pulp as p
import matplotlib.pyplot as plt
import numpy as np

import pywt
import pywt.data

import cv2

from statistics import NormalDist
from scipy.stats import beta

#cost function
def c(x,y):
  #return np.sqrt(x**2 + y**2)
  #return 4*y*x**2-x*y**2
  return -x*y**2 + y*x**2
  #return ((2*y-x-1)**2)*((2*y-x))**2

def restri1(Lambda, k, n):
  sum = 0
  for j in range(2**n):
    sum = sum + Lambda[k][j]
  return sum

def restri2(Lambda, j, n):
  sum = 0
  for k in range(2**n):
    sum = sum + Lambda[k][j]
  return sum

def funcion_objetivo(Lambda, X, Y):
  sum = 0
  for i in range(len(X)):
    for j in range(len(Y)):
      sum = sum + Lambda[i][j] * c(X[i], Y[j])
  return sum

def funcion_objetivo1(Lambda, Ind1, X1, Y1):
  sum = 0
  for k in range(len(Ind1)):
    sum = sum + Lambda[Ind1[k][0]][Ind1[k][1]] * c(X1[Ind1[k][0]], Y1[Ind1[k][1]])
  return sum

def truncNormal1(x,y):
  A = NormalDist(mu=0.5, sigma=1).cdf(1) - NormalDist(mu=0.5, sigma=1).cdf(0)
  return (NormalDist(mu=0.5, sigma=1).cdf(y) - NormalDist(mu=0.5, sigma=1).cdf(x))/A

def cumlBeta1(x,y):
  return beta.cdf(y, 2, 2) - beta.cdf(x, 2, 2)



n = 6
k1 = 2

X = []
Y = []

for i in range(1, 2**n + 1):
  X.append(i/2**n)
  Y.append(i/2**n)

Lambda = []
for i in range(2**n):
  lam = []
  for j in range(2**n):
    lam.append(j)
  Lambda.append(lam)


model = p.LpProblem("Modelo", p.LpMinimize)

for i in range(len(X)):
  for j in range(len(Y)):
    Lambda[i][j] = p.LpVariable(f"Lambda_{i}_{j}", lowBound=0)



model += funcion_objetivo(Lambda, X, Y)

#rest
for k in range(2**n):
  #model += restri1(Lambda, k, n) == 1/(2**n)
  model += restri1(Lambda, k, n) == truncNormal1((k)/(2**n), (k+1)/(2**n))

for j in range(2**n):
  model += restri2(Lambda, j, n) == 1/(2**n)

model.solve()

# opt value
#model.objective.value()

c0 = np.zeros((len(X), len(Y)))
for i in range(len(X)):
  for j in range(len(Y)):
    if Lambda[i][j].varValue != 0:
      c0[i][j] = 255
cv2.imwrite(f"mass_transfer_n={n}.png", c0)
#cv2.imwrite(f"mass_transfer2_n={n}.png", c0)

###
Value = [model.objective.value()]
###
Variables = [2**(2*n)]
###

l = 0
X_l = X.copy()
Y_l = Y.copy()
Lambda_l = Lambda.copy()


while l<k1:
  #wavelet implementation
  co_l = np.zeros((len(X_l), len(Y_l)))
  for i in range(len(X_l)):
    for j in range(len(Y_l)):
      if Lambda_l[i][j] != 0:
        if Lambda_l[i][j].varValue != 0:
          co_l[i][j] = Lambda_l[i][j].varValue

  coeffs_l = pywt.dwt2(co_l, 'coif1')
  LL_l, (LH_l, HL_l, HH_l) = coeffs_l

  ########

  Lambda_l = []
  X_l = []
  Y_l = []

  for i in range(1, 2**(n+1+l) + 1):
    X_l.append(i/2**(n+1+l))
    Y_l.append(i/2**(n+1+l))

  for i in range(4*(np.shape(coeffs_l[1][2])[0] -2)):
    lam = []
    for j in range(4*(np.shape(coeffs_l[1][2])[1] -2)):
      lam.append(0.0)
    Lambda_l.append(lam)

  for i in range(1, np.shape(coeffs_l[1][2])[0]-1):
    for j in range(1, np.shape(coeffs_l[1][2])[1]-1):
      if coeffs_l[1][2][i][j]!= 0:
        for k in range(4):
          for m in range(4):
            Lambda_l[4*(i-1)+k][4*(j-1)+m] = 1

  ###### Graph Lambda

  fig, ax = plt.subplots(figsize=(10, 10))
  img = ax.imshow(np.rot90(Lambda_l), interpolation="nearest", cmap=plt.cm.inferno)
  #fig.colorbar(img)
  ax.set_title(f"Lambda_({n}+{l+1})")
  plt.savefig(f"Coif1_Lambda4_{n + l + 1}_graph.png")
  #plt.savefig(f"Coif1_f2Lambda2_{n + l + 1}_graph.png")
  plt.close(fig) # Close the plot to free up memory

  ## index of non zero variables
  Ind_l = []
  for i in range(len(X_l)):
    for j in range(len(Y_l)):
      if Lambda_l[i][j] == 1.0:
        Ind_l.append([i,j])

  ####
  Variables.append(len(Ind_l))
  ####

  #LpProblem

  model_l = p.LpProblem(f"Model({n}+{l+1})", p.LpMinimize)

  for k in range(len(Ind_l)):
    Lambda_l[Ind_l[k][0]][Ind_l[k][1]] = p.LpVariable(f"Lambda_{Ind_l[k][0]}_{Ind_l[k][1]}", lowBound=0)

  model_l += funcion_objetivo1(Lambda_l, Ind_l, X_l, Y_l)

  #restrictions
  for k in range(2**(n+l+1)):
    #model_l += restri1(Lambda_l, k, n+l+1) == 1/(2**(n+l+1))
    model_l += restri1(Lambda_l, k, n+l+1) == truncNormal1((k)/(2**(n+l+1)), (k+1)/(2**(n+l+1)))

  for j in range(2**(n+l+1)):
    model_l += restri2(Lambda_l, j, n+l+1) == 1/(2**(n+l+1))

  model_l.solve()

  #####
  Value.append(model_l.objective.value())
  #####

  #save transference plan
  c0_l = np.zeros((len(X_l), len(Y_l)))
  for i in range(len(X_l)):
    for j in range(len(Y_l)):
      if Lambda_l[i][j] != 0:
        if Lambda_l[i][j].varValue != 0:
          c0_l[i][j] = 255
  cv2.imwrite(f"Coif1_L1_mass_transfer4_n={n+l+1}.png", c0_l)
  #cv2.imwrite(f"Coif1_L1_mass_transfer2_n={n+l+1}.png", c0_l)

  l = l+1


print(Value)
print(Variables)