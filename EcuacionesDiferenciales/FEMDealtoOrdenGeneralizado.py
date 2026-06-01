import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
#%% Funciones necesarias
def Nm(order, x, local_node, Mnod, Melem, e):
    x = np.asarray(x, dtype=float)
    nods = Melem[e]
    x_nodes = Mnod[nods]
    xj = x_nodes[local_node]
    L = np.ones_like(x, dtype=float)
    for m in range(order+1):
        if m != local_node:
            L *= (x - x_nodes[m])/(xj - x_nodes[m])
    return L
def dNm(order, x, local_node, Mnod, Melem, e):
    x = np.asarray(x, dtype=float)
    nods = Melem[e]
    x_nodes = Mnod[nods]
    xj = x_nodes[local_node]
    dL = np.zeros_like(x, dtype=float)
    for m in range(order+1):
        if m == local_node:
            continue
        term = np.ones_like(x, dtype=float) * (1.0/(xj - x_nodes[m]))
        for k in range(order+1):
            if k != local_node and k != m:
                term *= (x - x_nodes[k])/(xj - x_nodes[k])
        dL += term
    return dL
def gauss_legendre_integral(func, a, b, quad_n):
    xi, wi = leggauss(quad_n)   
    xm = 0.5*(b-a)*xi + 0.5*(b+a)   
    wm = 0.5*(b-a)*wi               
    y = func(xm)
    y = np.asarray(y)
    y = y.reshape((xi.size, -1)) if y.ndim > 1 else y.ravel()
    if y.ndim == 1:
        result = np.dot(wm, y)
    else:
        reduced = np.dot(wm, y)
        result = reduced.sum()
    return float(result)
def generate_mesh_1d(a, b, Ne, order):
    Mnod = []
    Melem = []
    h = (b - a) / Ne
    p = order
    nodes_per_elem = p + 1
    x_loc = np.linspace(0, h, nodes_per_elem)
    current = 0
    for e in range(Ne):
        x_left = a + e*h
        x_elem = x_left + x_loc
        if e == 0:
            for x in x_elem:
                Mnod.append(x)
            Melem.append(list(range(nodes_per_elem)))
            current = nodes_per_elem
        else:
            for x in x_elem[1:]:
                Mnod.append(x)
            Melem.append(list(range(current-1, current+p)))
            current += p

    return np.array(Mnod), np.array(Melem, dtype=int)
def plot_mesh_1D(Mnod, Melem, title="Malla 1D"):
    """
    Plotea una malla 1D e indica en rojo los nodos exteriores de cada elemento.
    
    Mnod  : arreglo de coordenadas nodales (Nnod,)
    Melem : conectividades (Ne, nen)
    """
    plt.figure(figsize=(10, 1.5))
    y = np.zeros_like(Mnod)
    # Graficar nodos y líneas de elementos
    plt.plot(Mnod, y, 'ko-', ms=8)
    # Etiquetas de nodos
    for i, x in enumerate(Mnod):
        plt.text(x, 0.02, f"N{i}", ha='center')

    # Etiquetas de elementos + identificación de nodos exteriores
    outer_nodes = set()   # para evitar duplicar puntos rojos

    for e, conn in enumerate(Melem):
        # centro geométrico del elemento
        x_mid = np.mean(Mnod[conn])
        plt.text(x_mid, -0.03, f"E{e}", ha='center', color='blue')

        # nodos exteriores: primero y último del elemento
        outer_nodes.add(conn[0])
        outer_nodes.add(conn[-1])

    # Ploteo de nodos exteriores en rojo
    for n in outer_nodes:
        plt.plot(Mnod[n], 0, 'ro', ms=10)

    plt.title(title)
    plt.yticks([])
    plt.xlabel("x")
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.ylim(-0.1, 0.1)
    plt.show()
def K_e(Mnod,Melem, e,order, k_dif=0.1,k_advec=0, k_reac=10.0,quad_n=2):
    nods=Melem[e,:]
    xi=Mnod[nods[0]]
    xf=Mnod[nods[-1]]
    if quad_n is None:
        quad_n = order + 1
    ndofs = len(nods)
    Ke_diff = np.zeros((ndofs,ndofs),dtype=np.float64)
    Ke_advec = np.zeros((ndofs,ndofs),dtype=np.float64)
    Ke_reac = np.zeros((ndofs,ndofs),dtype=np.float64)
    for i_local in range(ndofs):
        for j_local in range(ndofs):
            integrand_diff = lambda x: k_dif * dNm(order,x, i_local, Mnod, Melem, e) * dNm(order,x, j_local, Mnod, Melem, e)
            integrand_advec = lambda x: k_advec * Nm(order,x, i_local, Mnod, Melem, e) * dNm(order,x, j_local, Mnod, Melem, e)
            integrand_reac = lambda x: k_reac * Nm(order,x, i_local, Mnod, Melem, e) * Nm(order,x, j_local, Mnod, Melem, e)
            Ke_diff[i_local, j_local] = gauss_legendre_integral(integrand_diff, xi, xf, quad_n)
            Ke_advec[i_local, j_local] = gauss_legendre_integral(integrand_advec, xi, xf, quad_n)
            Ke_reac[i_local, j_local] = gauss_legendre_integral(integrand_reac, xi, xf, quad_n)
    # print (Ke_diff + Ke_advec + Ke_reac)
    return Ke_diff + Ke_advec + Ke_reac
def F_e(Mnod, Melem, e,order, f_func, quad_n=2):
    nods=Melem[e,:]
    xi=Mnod[nods[0]]
    xf=Mnod[nods[-1]]
    if quad_n is None:
        quad_n = order + 1
    ndofs = len(nods)
    fe = np.zeros((ndofs,1),dtype=np.float64)
    for i_local in range(ndofs):
        integrand = lambda x: f_func(x) * Nm(order,x, i_local, Mnod, Melem, e)
        fe[i_local,0] = gauss_legendre_integral(integrand, xi, xf, quad_n)
        # print(fe)
    return fe
def assemble_KG(Mnod, Melem,order, f_func, k_dif=0.1,k_advec=0.0,k_reac=10.0, quad_n=2):
    n_nodes = Mnod.shape[0]
    KG = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    FG = np.zeros(n_nodes, dtype=np.float64)
    n_elems = Melem.shape[0]
    for e in range(n_elems):
        dofs = Melem[e,:]
        Ke = K_e(Mnod, Melem, e,order, k_dif=k_dif,k_advec=k_advec, k_reac=k_reac, quad_n=quad_n)
        Fe = F_e(Mnod, Melem, e,order, f_func, quad_n=quad_n)
        for i in range(len(dofs)):
            for j in range (len(dofs)):
                KG[dofs[i], dofs[j]] += Ke[i, j]
            FG[dofs[i]] += Fe[i]
    return KG, FG
def penalizacion(K, F, dirichlet={}, neumann={}, penalty=1e20):
    K_mod = K.copy()
    F_mod = F.copy()
    # --- Condiciones de Neumann (fuerzas o flujos) ---
    for node, value in neumann.items():
        F_mod[node] += value
    # --- Condiciones de Dirichlet (penalty) ---
    for node, value in dirichlet.items():
        K_mod[node, node] += penalty
        F_mod[node] += penalty * value
    return K_mod, F_mod
#%% Solucionar el caso
N_elems = 4
a = 0
b = 1
order = 20
Mnod,Melem = generate_mesh_1d(a, b, N_elems,order)
plot_mesh_1D(Mnod, Melem)
f_func = lambda x: x**3
# Ensamble
KG, FG = assemble_KG(Mnod, Melem,order, f_func, k_dif=0.1,k_advec=0.0, k_reac=10.0, quad_n=order+1)
plt.figure(figsize=(5,4))
plt.spy(KG)
plt.title("Matriz de rigidez global KG ")
plt.tight_layout()
kappa = 1e20
KG[0,0]+=kappa
KG[-1,-1]+=kappa
FG[0]+=kappa*0
FG[-1]+=kappa*0
sol = np.linalg.solve(KG, FG)
X = np.linspace(0, 1, 400)
phi_approx = np.zeros_like(X)
elem_index = np.minimum((X * N_elems).astype(int), N_elems - 1)
for idx, x in enumerate(X):
    e = elem_index[idx]
    nodes = Melem[e]
    nn_loc = order + 1
    Nvals = np.array([Nm(order, x, a, Mnod, Melem, e) for a in range(nn_loc)])
    phi_approx[idx] = sol[nodes].dot(Nvals)
def f_exacta(x):
    return -(np.exp(-10*x)*(-53*np.exp(10)+((50-50*np.exp(20))*x**3+(3-3*np.exp(20))*x)*np.exp(10*x)+53*np.exp(10+20*x)))/(500*np.exp(20)-500)
plt.figure()
plt.scatter(Mnod, sol,color="blue", label='puntos calculados')
plt.plot(X, phi_approx, "b",lw=1.2, label='FEM lineal')
plt.plot(X, f_exacta(X), 'k--', lw=1.2, label='Solución exacta')
plt.legend()
plt.xlabel('x')
plt.ylabel(r'$\phi(x)$')
plt.title('FEM 1D - elementos lineales')
plt.grid(True)
plt.show()
#%%
orders = [1, 2, 3, 4]
Ne_list = np.unique(np.int64(np.logspace(0.5, 3, 10)))  
# 10 valores entre ~3 y 1000 elementos
f_func = lambda x: x**3
# --- malla de referencia ---
X = np.linspace(0, 1, 2000)
phi_ex = f_exacta(X)
plt.figure(figsize=(8,6))
# guardamos errores para graficar al final
errores_por_p = {}
hs_por_p = {}
for order in orders:
    errores_L2 = []
    hs = []
    print(f"\nProcesando orden p = {order} ...")
    for Ne in Ne_list:
        # ===== generar malla para este orden =====
        Mnod, Melem = generate_mesh_1d(0, 1, Ne, order)
        # ===== ensamblaje =====
        KG, FG = assemble_KG(Mnod, Melem, order,
                             f_func,
                             k_dif=0.1,
                             k_advec=0.0,
                             k_reac=10.0,
                             quad_n=order+1)

        # ===== condiciones de borde =====
        KG[0,0] += 1e20
        FG[0]   += 0
        KG[-1,-1] += 1e20
        FG[-1]    += 0
        # ===== resolver =====
        sol = np.linalg.solve(KG, FG)
        # ===== evaluar FEM en X =====
        phi_approx = np.zeros_like(X)
        # buscar elemento al cual pertenece cada punto
        elem_index = np.minimum((X * Ne).astype(int), Ne-1)
        for idx, x in enumerate(X):
            e = elem_index[idx]
            nodes = Melem[e]
            nn_loc = order + 1
            Nvals = np.array([Nm(order, x, a, Mnod, Melem, e)
                              for a in range(nn_loc)])
            phi_approx[idx] = sol[nodes].dot(Nvals)
        # ===== error L2 =====
        error_L2 = np.sqrt(np.trapz((phi_ex - phi_approx)**2, X))
        errores_L2.append(error_L2)
        hs.append(1/Ne)
    # guardamos
    errores_por_p[order] = errores_L2
    hs_por_p[order] = hs
    # graficar curvas
    plt.loglog(hs, errores_L2, 'o-', lw=1, label=f"p = {order}")
# ===== gráfica final =====
plt.gca().invert_xaxis()
plt.xlabel("h = 1 / N_e")
plt.ylabel(r"$\|e\|_{L^2}$")
plt.title("Convergencia FEM 1D — h-refinamiento para p = 1..4")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()
#%%
L = 1        # [m]
hc = 9.0       # [W/m^2/C]
k = 360.0      # [W/m/C]
A = 0.001      # [m^2]
p = 2.002      # [m]
Tc = 235.0     # [C]
Tinf = 20.0    # [C]
alpha = (hc*p)/(k*A)
N_elems = 3
a = 0
b = L
order = 2
Mnod,Melem = generate_mesh_1d(a, b, N_elems,order)
plot_mesh_1D(Mnod, Melem)
f_func = lambda x: alpha*Tinf
# Ensamble
KG, FG = assemble_KG(Mnod, Melem,order, f_func, k_dif=1,k_advec=0.0, k_reac=alpha, quad_n=order+1)
plt.figure(figsize=(5,4))
plt.spy(KG)
plt.title("Matriz de rigidez global KG ")
kappa = 1e20
KG[0,0]+=kappa
FG[0]+=kappa*Tc
FG[-1]+=0 
sol = np.linalg.solve(KG, FG)
X = np.linspace(0, L, 400)
phi_approx = np.zeros_like(X)
elem_index = np.minimum((X * N_elems).astype(int), N_elems - 1)
for idx, x in enumerate(X):
    e = elem_index[idx]
    nodes = Melem[e]
    nn_loc = order + 1
    Nvals = np.array([Nm(order, x, a, Mnod, Melem, e) for a in range(nn_loc)])
    phi_approx[idx] = sol[nodes].dot(Nvals)
C1 = np.sqrt(alpha)
def f_exacta(x):
    return np.exp(-C1*x) * (
        np.exp(2*C1*L)*Tc
        + np.exp(2*C1*x)*Tc
        - np.exp(2*C1*L)*Tinf
        + np.exp(C1*x)*Tinf
        - np.exp(2*C1*x)*Tinf
        + np.exp(2*C1*L + C1*x)*Tinf
    ) / (1 + np.exp(2*C1*L))
plt.figure()
plt.scatter(Mnod, sol,color="blue", label='puntos calculados')
plt.plot(X, phi_approx, "b",lw=1.2, label='FEM lineal')
plt.plot(X, f_exacta(X), 'k--', lw=1.2, label='Solución exacta')
plt.legend()
plt.xlabel('x')
plt.ylabel(r'$\phi(x)$')
plt.title('FEM 1D - elementos lineales')
plt.grid(True)
plt.show()
#%%
L = 1        # [m]
hc = 9.0       # [W/m^2/C]
k = 360.0      # [W/m/C]
A = 0.001      # [m^2]
p = 2.002      # [m]
Tc = 235.0     # [C]
Tinf = 20.0    # [C]
alpha = (hc*p)/(k*A)
N_elems = 5
a = 0
b = L
order = 10
Mnod,Melem = generate_mesh_1d(a, b, N_elems,order)
plot_mesh_1D(Mnod, Melem)
f_func = lambda x: 1
# Ensamble
KG, FG = assemble_KG(Mnod, Melem,order, f_func, k_dif=1,k_advec=3, k_reac=0, quad_n=order+1)
plt.figure(figsize=(5,4))
plt.spy(KG)
plt.title("Matriz de rigidez global KG ")
kappa = 1e20
KG[0,0]+=kappa
FG[0]+=kappa*0
KG[-1,-1]+=kappa
FG[-1]+=kappa*0
sol = np.linalg.solve(KG, FG)
X = np.linspace(0, L, 400)
phi_approx = np.zeros_like(X)
elem_index = np.minimum((X * N_elems).astype(int), N_elems - 1)
for idx, x in enumerate(X):
    e = elem_index[idx]
    nodes = Melem[e]
    nn_loc = order + 1
    Nvals = np.array([Nm(order, x, a, Mnod, Melem, e) for a in range(nn_loc)])
    phi_approx[idx] = sol[nodes].dot(Nvals)
plt.figure()
plt.scatter(Mnod, sol,color="blue", label='puntos calculados')
plt.plot(X, phi_approx, "b",lw=1.2, label='FEM lineal')
# plt.plot(X, f_exacta(X), 'k--', lw=1.2, label='Solución exacta')
plt.legend()
plt.xlabel('x')
plt.ylabel(r'$\phi(x)$')
plt.title('FEM 1D - elementos lineales')
plt.grid(True)
plt.show()
#%%
L = 1        # [m]
hc = 9.0       # [W/m^2/C]
k = 360.0      # [W/m/C]
A = 0.001      # [m^2]
p = 2.002      # [m]
Tc = 235.0     # [C]
Tinf = 20.0    # [C]
alpha = (hc*p)/(k*A)
N_elems = 5
a = 0
b = L
order = 1
Mnod,Melem = generate_mesh_1d(a, b, N_elems,order)
plot_mesh_1D(Mnod, Melem)
f_func = lambda x: 1
# Ensamble
KG, FG = assemble_KG(Mnod, Melem,order, f_func, k_dif=1,k_advec=3, k_reac=0, quad_n=order+1)
plt.figure(figsize=(5,4))
plt.spy(KG)
plt.title("Matriz de rigidez global KG ")
kappa = 1e20
KG[0,0]+=kappa
FG[0]+=kappa*0
FG[-1]+=-1
sol = np.linalg.solve(KG, FG)
X = np.linspace(0, L, 400)
phi_approx = np.zeros_like(X)
elem_index = np.minimum((X * N_elems).astype(int), N_elems - 1)
for idx, x in enumerate(X):
    e = elem_index[idx]
    nodes = Melem[e]
    nn_loc = order + 1
    Nvals = np.array([Nm(order, x, a, Mnod, Melem, e) for a in range(nn_loc)])
    phi_approx[idx] = sol[nodes].dot(Nvals)
plt.figure()
plt.scatter(Mnod, sol,color="blue", label='puntos calculados')
plt.plot(X, phi_approx, "b",lw=1.2, label='FEM lineal')
# plt.plot(X, f_exacta(X), 'k--', lw=1.2, label='Solución exacta')
plt.legend()
plt.xlabel('x')
plt.ylabel(r'$\phi(x)$')
plt.title('FEM 1D - elementos lineales')
plt.grid(True)
plt.show()
#%%
L = 1        # [m]
hc = 9.0       # [W/m^2/C]
k = 360.0      # [W/m/C]
A = 0.001      # [m^2]
p = 2.002      # [m]
Tc = 235.0     # [C]
Tinf = 20.0    # [C]
alpha = (hc*p)/(k*A)
N_elems = 5
a = 0
b = L
order = 1
Mnod,Melem = generate_mesh_1d(a, b, N_elems,order)
plot_mesh_1D(Mnod, Melem)
f_func = lambda x: 1
# Ensamble
KG, FG = assemble_KG(Mnod, Melem,order, f_func, k_dif=1,k_advec=3, k_reac=0, quad_n=order+1)
plt.figure(figsize=(5,4))
plt.spy(KG)
plt.title("Matriz de rigidez global KG ")
kappa = 1e20
KG[0,0]+=kappa
FG[0]+=kappa*0
KG[-1,-1] +=0.5 
FG[-1]+=10
sol = np.linalg.solve(KG, FG)
X = np.linspace(0, L, 400)
phi_approx = np.zeros_like(X)
elem_index = np.minimum((X * N_elems).astype(int), N_elems - 1)
for idx, x in enumerate(X):
    e = elem_index[idx]
    nodes = Melem[e]
    nn_loc = order + 1
    Nvals = np.array([Nm(order, x, a, Mnod, Melem, e) for a in range(nn_loc)])
    phi_approx[idx] = sol[nodes].dot(Nvals)
plt.figure()
plt.scatter(Mnod, sol,color="blue", label='puntos calculados')
plt.plot(X, phi_approx, "b",lw=1.2, label='FEM lineal')
# plt.plot(X, f_exacta(X), 'k--', lw=1.2, label='Solución exacta')
plt.legend()
plt.xlabel('x')
plt.ylabel(r'$\phi(x)$')
plt.title('FEM 1D - elementos lineales')
plt.grid(True)
plt.show()
#%%
N_elems = 20
a = 0
b = 100
order = 1
Mnod,Melem = generate_mesh_1d(a, b, N_elems,order)
plot_mesh_1D(Mnod, Melem)
f_func = lambda x: 1
# Ensamble
KG, FG = assemble_KG(Mnod, Melem,order, f_func, k_dif=1.4,k_advec=-0.03, k_reac=-1/2000, quad_n=order+1)
plt.figure(figsize=(5,4))
plt.spy(KG)
plt.title("Matriz de rigidez global KG ")
kappa = 1e20
U = 0.03
c_in =100e3
D = 1.4
KG[0,0]+= U/D
FG[0] += -(U*c_in)/D
FG[-1]+=0
sol = np.linalg.solve(KG, FG)
X = np.linspace(0, b, 400)
phi_approx = np.zeros_like(X)
elem_index = np.minimum((X * N_elems).astype(int), N_elems - 1)
for idx, x in enumerate(X):
    e = elem_index[idx]
    nodes = Melem[e]
    nn_loc = order + 1
    Nvals = np.array([Nm(order, x, a, Mnod, Melem, e) for a in range(nn_loc)])
    phi_approx[idx] = sol[nodes].dot(Nvals)
plt.figure()
plt.scatter(Mnod, sol,color="blue", label='puntos calculados')
# plt.plot(X, phi_approx, "b",lw=1.2, label='FEM lineal')
# plt.plot(X, f_exacta(X), 'k--', lw=1.2, label='Solución exacta')
plt.legend()
plt.xlabel('x')
plt.ylabel(r'$\phi(x)$')
plt.title('FEM 1D - elementos lineales')
plt.grid(True)
plt.show()