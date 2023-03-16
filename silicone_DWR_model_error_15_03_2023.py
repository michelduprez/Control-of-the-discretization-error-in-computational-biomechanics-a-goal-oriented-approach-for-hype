# command for fenics in the terminal
#sudo docker run -ti -v $(pwd):/home/fenics/shared:z quay.io/fenicsproject/stable
#
from __future__ import print_function
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import pygmsh
from mpl_toolkits.mplot3d import Axes3D
# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
if has_linear_algebra_backend("Epetra"):
        parameters["linear_algebra_backend"] = "Epetra"
parameters["allow_extrapolation"] = True # for the refinement of the mesh
parameters["refinement_algorithm"] = "plaza_with_parent_facets" 



#######################
#### PARAMETERS
#######################

# maximum of iterations of adaptation
max_ite = 7


# Choice of the function of interest
#if interest ==1:
#	J(v) = force bord superieur (mesh with 5 holes)
#if interest ==2:
#	J(v) = L2 norm on the interior rectangular (mesh without hole)
#if interest ==3:
#	J(v) = H1 norm on the interior rectangular (mesh without hole)
#if interest ==4:
#	J(v) = trace of green lagrange interior rectangle (mesh without hole)
#if interest ==5:
#	J(v) = vonmisesstress interior rectangle (mesh without hole)

interest = 1


# refinement fraction in the dorfler strategy of DWR
refinement_fraction = 0.3

# Adaptive or uniform refiement 
uniform = False
#uniform= True


# Choice of the estimator
#print("Approximation of z-phi_h in eta_T :")
#print("1) E(z_h^1) - z_h^1")
#print("2) E(z_h^1) - I_h(E(z_h^1))")
#print("3) z_h^2 - I_hz_h^2")
#print("4) z_h^2 - z_h^1")
estim = 2


# order of the polynomial in the finite element space for the displacement
order = 2


# size of the cells in the intial mesh
cell_size = 5.0


# number of iteration in the resolution of the non linear problem
nb_iter_chargement = 20


### choise of law
# 1: MooneyRivlin
# 2: Gent
# 3: Haines and Wilson
FirstPiolaKirchhoffStress_Compressible = 2


# degree of polynomial approximation
polV = 2

# prefixe of the file
nomFic = 'output_silicone/'
if uniform == True:
    nomFic += 'uniform_'
else:
    nomFic += 'adaptative_'
nomFic += '_interest_{name}/'.format(name=interest)	
if FirstPiolaKirchhoffStress_Compressible == 1:
    nomFic += '_mooney'
if FirstPiolaKirchhoffStress_Compressible == 2:
    nomFic += '_gent'
if FirstPiolaKirchhoffStress_Compressible == 3:
    nomFic += '_uniform_wilson'
nomFic += '_ite_{name}/'.format(name=max_ite)		


# parameters of the solver
iterative_solver = True
list_linear_solver_methods()
list_krylov_solver_preconditioners()


# for boundaries
NEUMANN_BOUNDARY = 1
DIRICHLET_BOUNDARY = 2
NEUMANN_BOUNDARY_NO_LOAD = 3
GREEN_LAGRANGE = 4
INF_L0 = 4
SUP_L0 = 5


def write_to_file(f,A,B):
    f.write('\n')
    for i in range(len(A)):
        f.write('(')
        f.write(str(A[i]))
        f.write(',')
        f.write(str(B[i]))
        f.write(')')
        f.write('\n')
    f.write('\n')
    f.write('\n')
    return;


def DeformationGradient(u):
    d = u.geometric_dimension()
    I = Identity(d)             # Identity tensor
    F = I + grad(u)             # Deformation gradient
    return F


def GreenLagrange(u):
    d = u.geometric_dimension()
    I = Identity(d)             # Identity tensor
    F = I + grad(u)             # Deformation gradient
    C = F.T*F  
    return 0.5*(C-I)


def VonMisesStress(P):
    return sqrt(0.5*( (P[0,0]-P[1,1] )**2 + (P[1,1]-P[2,2] )**2 + (P[2,2]-P[0,0] )**2 +6*(P[0,1]*P[0,1] + P[1,2]*P[1,2] + P[2,0]*P[2,0] ) ) )

#######################
#### hyperelastic laws
#######################

# Mooney-Rivlin material coefficients
if FirstPiolaKirchhoffStress_Compressible == 1:
    c10 = 0.14 # MPa
    c01 = 0.023 # MPa
    print('lois : Mooney')
    print('c10 :',float(c10))
    print('c01 :',float(c01))


# Gent material coefficients
if FirstPiolaKirchhoffStress_Compressible == 2:
    E = Constant(0.97)
    Jm = Constant(13.0)
    print('lois : Gent')
    print('E :',float(E))
    print('Jm :',float(Jm))



# Haines_Wilson material coefficients
if FirstPiolaKirchhoffStress_Compressible == 3:
    c10 = Constant(0.14)
    c01 = Constant(0.033)
    c20 = Constant(-0.0026)
    c02 = Constant(0.00095)
    c30 = Constant(0.0038)
    c11 = Constant(- 0.0049)
    print('lois : Haines_Wilson')
    print('c10 :',float(c10))
    print('c01 :',float(c01))
    print('c20 :',float(c20))
    print('c02 :',float(c02))
    print('c30 :',float(c30))
    print('c11 :',float(c11))


def FirstPiolaKirchhoffStress_Compressible_Mooney_Rivlin(c10,c01,u,p):
    d = u.geometric_dimension()
    I = Identity(d)             # Identity tensor
    F = variable(I + grad(u))            # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor
    J = det(F)
    I1 = tr(C)
    I2 = 0.5*(tr(C)**2-tr(C*C))
    J1 = I1*J**(-2/3.0)
    J2 = I2*J**(-4/3.0)
    W = c10*(J1-3.0) + c01*(J2-3.0) - p*(det(C)-1.0)
    P = diff(W,F)
    return P


def FirstPiolaKirchhoffStress_Compressible_Gent(E,Jm,u,p):
    d = u.geometric_dimension()
    I = Identity(d)             # Identity tensor
    F = variable(I + grad(u))            # Deformation gradient
    C = variable(F.T*F)                   # Right Cauchy-Green tensor
    J = variable(det(F))
    I1 = variable(tr(C))
    J1 = variable(I1*J**(-2/3.0))
    W = (-E/6.0)*Jm*ln(1.0-(J1-3.0)/Jm)  - p*(det(C)-1.0)
    P = diff(W,F)
    return P



def FirstPiolaKirchhoffStress_Compressible_Haines_Wilson(c10,c01,c20,c02,c30,c11,u,p):
    d = u.geometric_dimension()
    I = Identity(d)             # Identity tensor
    F = variable(I + grad(u))            # Deformation gradient
    C = variable(F.T*F)                   # Right Cauchy-Green tensor
    J = variable(det(F))
    I1 = variable(tr(C))
    I2 = variable(0.5*(tr(C)**2-tr(C*C)))
    J1 = variable(I1*J**(-2/3.0))
    J2 = variable(I2*J**(-4/3.0))
    W = c10*(J1-d) + c01*(J2-d) + c20*(J1-d)**2 + c02*(J2-d)**2 + c30*(J1-d)**3 + c11*(J1-d)*(J2-d)  - p*(det(C)-1.0)
    P = diff(W,F)
    return P



#######################
#### boundary conditions and omega
#######################

class NeumannBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1],0.0)

class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1],-82.5)

class Neumann_No_Load_Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class Omega(SubDomain):
    def inside(self, x, on_boundary):
        return x[0]>3.0*largeur/4.0 and x[0]<largeur/4.0 and x[1]>3.0*hauteur/4.0 and x[1]<hauteur/4.0
     
        
#######################
#### Refinement of the mesh from the indicator
#######################

def refinement(eta, mesh):
	eta_ind = np.concatenate((np.array([eta]),np.array([range(len(eta))])),axis=0)
	eta_ind = eta_ind.T[np.lexsort(np.fliplr(eta_ind.T).T)].T
	eta_ind = eta_ind[:,::-1]
	cell_markers = MeshFunction("bool", mesh,mesh.topology().dim())
	cell_markers.set_all(False)
	ind=0
	while np.sum(eta_ind[0,:ind])<refinement_fraction*np.sum(eta):
		cell_markers[int(eta_ind[1,ind])]=True
		ind=ind+1
	return cell_markers


#######################
#### mesh
#######################

# Generate mesh
# position of the holes
C1x, C1y = -47.5, -21.4 # en y dans le papier c est 21.2
C2x, C2y = -14.0,-23.0
C3x, C3y = -31.5, -41.0 # en y dans le papier c est -59
C4x, C4y = -47.7, -59.1 # coordonnees manquantes dans le papier
C5x, C5y = -14.5, -58.0
rayon=10.0
eps = 0.2

#GMSH avec utilisation opencascade
#creation d une surface 2D que l on extrudera
with pygmsh.occ.Geometry() as geo:
    geo.characteristic_length_min=cell_size
    geo.characteristic_length_max=cell_size
    largeur= -61.5
    hauteur= -82.5
    plaquePleine = geo.add_rectangle([0.0, 0.0,0.0], largeur, hauteur)
    if interest ==1:
        fissure = geo.add_polygon(
        [
            [C1x + eps, C1y +eps,0. ],
            [C3x + eps, C3y +eps,0. ],
            [C3x - eps, C3y -eps,0. ],
            [C1x - eps, C1y -eps,0. ]
        ],
        mesh_size=1.,)
        disque1 = geo.add_disk([C1x, C1y,0.],rayon)
        disque2 = geo.add_disk([C2x, C2y,0.],rayon)
        disque3 = geo.add_disk([C3x, C3y,0.],rayon)
        disque4 = geo.add_disk([C4x, C4y,0.],rayon)
        disque5 = geo.add_disk([C5x, C5y,0.],rayon)
        plaque =geo.boolean_difference( plaquePleine, geo.boolean_union( [fissure,disque1,disque2,disque3,disque4,disque5] ) )
    if interest !=1:
        #petitePlaque = geo.add_rectangle([largeur/4.0, hauteur/4.0,0.0], largeur/2.0, hauteur/2.0)
        #plaque = geo.boolean_fragments(petitePlaque,plaquePleine)
        plaquePleine = geo.add_rectangle([0.0, 0.0,0.0], largeur, hauteur)
        petiteLargeur= largeur/2.0
        petiteHauteur= hauteur/2.0
        originePP=[largeur/4.0, hauteur/4.0,0.0]
        petitePlaque0 = geo.add_rectangle(originePP, petiteLargeur, petiteHauteur)
        
        
        fissure = geo.add_polygon(
        [
            [C1x + eps, C1y +eps,0. ],
            [C3x + eps, C3y +eps,0. ],
            [C3x - eps, C3y -eps,0. ],
            [C1x - eps, C1y -eps,0. ]
        ],
        mesh_size=1.,)
        fissure1 = geo.add_polygon(
        [
            [C1x + eps, C1y +eps,0. ],
            [C3x + eps, C3y +eps,0. ],
            [C3x - eps, C3y -eps,0. ],
            [C1x - eps, C1y -eps,0. ]
        ],
        mesh_size=1.,)
        disque1 = geo.add_disk([C1x, C1y,0.],rayon)
        disque11 = geo.add_disk([C1x, C1y,0.],rayon)
        disque2 = geo.add_disk([C2x, C2y,0.],rayon)
        disque21 = geo.add_disk([C2x, C2y,0.],rayon)
        disque3 = geo.add_disk([C3x, C3y,0.],rayon)
        disque31 = geo.add_disk([C3x, C3y,0.],rayon)
        disque4 = geo.add_disk([C4x, C4y,0.],rayon)
        disque41 = geo.add_disk([C4x, C4y,0.],rayon)
        disque5 = geo.add_disk([C5x, C5y,0.],rayon)
        disque51 = geo.add_disk([C5x, C5y,0.],rayon)
        plaque2 =geo.boolean_difference( plaquePleine, geo.boolean_union( [fissure,disque1,disque2,disque3,disque4,disque5] ) )

        petitePlaque = geo.boolean_difference( petitePlaque0,geo.boolean_union([fissure1,disque11,disque21,disque31,disque41,disque51]))

        plaque = geo.boolean_fragments( petitePlaque,plaque2 )
    geo.extrude(plaque, [0,0,1.75] )
    meshGmsh = geo.generate_mesh()
    file_mesh = File(nomFic+"init_mesh.xml")
    meshGmsh.write(nomFic+"init_mesh.xml")
    mesh = Mesh(nomFic+'init_mesh.xml')

#save intial mesh
print('number of cells',mesh.num_cells())
vtkfile = File(nomFic+'init_mesh.pvd')
vtkfile << mesh

# Apres raffinement, projet les points des frontieres pour que les trous soient de forment circulaire
def ajust_hole(mesh):
    hole = Hole1()
    snap_hole_boundary(mesh,hole)
    hole = Hole2()
    snap_hole_boundary(mesh,hole)
    hole = Hole3()
    snap_hole_boundary(mesh,hole)
    hole = Hole4()
    snap_hole_boundary(mesh,hole)
    hole = Hole5()
    snap_hole_boundary(mesh,hole)


def snap_hole_boundary(mesh,subdomain):
    boundary = BoundaryMesh(mesh,"exterior")
    dim=mesh.geometry().dim()
    x = boundary.coordinates()
    for i in range (0,boundary.num_vertices()):
        subdomain.snap(x[i,:])
    ALE.move(mesh,boundary)
    
#implementation naive
toleranceRayon=1e-4
class Hole1(SubDomain):
    def inside(self, x, on_boundary):
        r = sqrt((x[0] - C1x)**2 + (x[1] - C1y)**2)
        return r <(1.+toleranceRayon)*rayon

    def snap(self, x):
        r = sqrt((x[0] - C1x)**2 + (x[1] - C1y)**2)
        if r < (1.+toleranceRayon)*rayon:
            x[0] = C1x + (rayon / r)*(x[0] - C1x)
            x[1] = C1y + (rayon / r)*(x[1] - C1y)

class Hole2(SubDomain):

    def inside(self, x, on_boundary):
        r = sqrt((x[0] - C2x)**2 + (x[1] - C2y)**2)
        return r < (1.+toleranceRayon)*rayon 

    def snap(self, x):
        r = sqrt((x[0] - C2x)**2 + (x[1] - C2y)**2)
        if r < (1.+toleranceRayon)*rayon:
            x[0] = C2x + (rayon / r)*(x[0] - C2x)
            x[1] = C2y + (rayon / r)*(x[1] - C2y)

class Hole3(SubDomain):

    def inside(self, x, on_boundary):
        r = sqrt((x[0] - C3x)**2 + (x[1] - C3y)**2)
        return r < (1.+toleranceRayon)*rayon 

    def snap(self, x):
        r = sqrt((x[0] - C3x)**2 + (x[1] - C3y)**2)
        if r < (1.+toleranceRayon)*rayon:
            x[0] = C3x + (rayon / r)*(x[0] - C3x)
            x[1] = C3y + (rayon / r)*(x[1] - C3y)

class Hole4(SubDomain):

    def inside(self, x, on_boundary):
        r = sqrt((x[0] - C4x)**2 + (x[1] - C4y)**2)
        return r < (1.+toleranceRayon)*rayon 

    def snap(self, x):
        r = sqrt((x[0] - C4x)**2 + (x[1] - C4y)**2)
        if r < (1.+toleranceRayon)*rayon:
            x[0] = C4x + (rayon / r)*(x[0] - C4x)
            x[1] = C4y + (rayon / r)*(x[1] - C4y)

class Hole5(SubDomain):

    def inside(self, x, on_boundary):
        r = sqrt((x[0] - C5x)**2 + (x[1] - C5y)**2)
        return r < (1.+toleranceRayon)*rayon 

    def snap(self, x):
        r = sqrt((x[0] - C5x)**2 + (x[1] - C5y)**2)
        if r < (1.+toleranceRayon)*rayon:
            x[0] = C5x + (rayon / r)*(x[0] - C5x)
            x[1] = C5y + (rayon / r)*(x[1] - C5y)



#######################
#### Print initial mesh
#######################
plt.figure()
ax=plt.axes(projection='3d')
plot(mesh, title=("Mesh Initial"))
ax.view_init(90,-90)


#######################
#### Print parameters
#######################
nbCellsmax= 1000000#16000
if uniform == True:
    nbCellsmax = 40000
    print("Uniform refinement")
else:
    nbCellsmax= 16000
    print("Adative refinement")
if interest==1:
    print('Quantity of interest : int_(upper boundary) P.n.n (mesh with 5 hole)')
if interest==2:
	print('Quantity of interest : L2 norm on the interior rectangular (mesh without hole)')
if interest==3:
	print('Quantity of interest : H1 norm on the interior rectangular (mesh without hole)')
if interest==4:
	print('Quantity of interest : green lagrange tensor on the interior rectangular (mesh without hole)')
if interest==5:
	print('Quantity of interest : vonmisestress on the interior rectangular (mesh without hole)')


# Used to limit the number of cells to ompute the "exact" solution
N_exact0 = 0
N_exact1 = 0
max_N_exact = 1e6

B  = Constant((0.0, 0.0,0.0))  # Body force per unit volume


#######################
#### # Initialization of array for the output
#######################
rEz_h = 1.0
init = True
ite = 0
Q_array = np.zeros(max_ite)
num_cell_array = np.zeros(max_ite)
sum_eta_T_array = np.zeros(max_ite)
eta_array = np.zeros(max_ite)
model_discretisation_error_array = np.zeros(max_ite)
discretisation_error_array1 = np.zeros(max_ite)
discretisation_error_array2 = np.zeros(max_ite)
eta_h_effi_array = np.zeros(max_ite)
sum_eta_T_effi_array = np.zeros(max_ite)
displac_array = np.zeros(max_ite)


#######################
#### Beginning of the adaptive algorithm
#######################
while ite < max_ite and (N_exact0 == 0 or 2*N_exact1 - N_exact0 < max_N_exact):
    print("##############################")
    print("ITERATION:",ite+1)
    if init == True:
        init = False
        meshRefine= Mesh(mesh)
    else:
        #tt = 1
        if uniform == False:
            meshRefine = refine(mesh, cell_markers)
            ajust_hole(meshRefine) 
        else:
            meshRefine = refine(mesh)
            ajust_hole(meshRefine) 

        
    mesh= Mesh(meshRefine)
    vtkfileIt = File(nomFic+'mesh_{}.pvd'.format(ite))
    vtkfileIt << mesh


    # Define boundary segments for Neumann conditions
    domain = MeshFunction("size_t", mesh, mesh.geometric_dimension(),0)
    omega = Omega()
    omega.mark(domain,1)
    
    boundaries = MeshFunction("size_t", mesh, mesh.geometric_dimension()-1)
    boundaries.set_all(NEUMANN_BOUNDARY_NO_LOAD)

    neumann_boundary = NeumannBoundary()
    neumann_boundary.mark(boundaries, NEUMANN_BOUNDARY)

    dirichlet_boundary = DirichletBoundary()
    dirichlet_boundary.mark(boundaries, DIRICHLET_BOUNDARY)


    # normal and size of the cells
    n = FacetNormal(mesh) 
    h = CellDiameter(mesh)
    print("Number of cells :",mesh.num_cells())	

    # Construction of the different spaces
    DG0 = FunctionSpace(mesh,'DG',0)
    P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = P2 * P1
    V = FunctionSpace(mesh, P2)
    VQ = FunctionSpace(mesh, TH)
    P3 = VectorElement("Lagrange", mesh.ufl_cell(), 3)
    P2 = FiniteElement("Lagrange", mesh.ufl_cell(), 2)
    TH2 = P3 * P2
    VQ2 = FunctionSpace(mesh, TH2)

    # Initialize cell function for domains
    dx = Measure("dx")(domain=mesh,subdomain_data = domain,metadata={'quadrature_degree': 5})
    ds = Measure("ds")(domain=mesh, subdomain_data = boundaries) # exterior facet
    dS = Measure("dS")()
    
    # Define functions
    dup = TrialFunction(VQ)            # Incremental displacement
    (v, q)  = TestFunctions(VQ)             # Test function
    up  = Function(VQ)                 # Displacement from previous iteration
    (u,p) = split(up)

    displacement_array = np.linspace(0.0,57.3,nb_iter_chargement)
    max_it=len(displacement_array)
    force_array = np.zeros(max_it)
    elongation_array = np.zeros(max_it)



    for rit in range(max_it):
        print('###########################################---------iteration: ', rit)
        displacement=displacement_array[rit]
        print('displacement',displacement)

        # Dirichlet boundary conditions
        Gamma_1 = DirichletBC(VQ.sub(0), Constant((0.0,0.0,0.0)), boundaries, DIRICHLET_BOUNDARY)
        Gamma_2 = DirichletBC(VQ.sub(0), Constant((0.0,displacement,0.0)), boundaries, NEUMANN_BOUNDARY)
        bcs = [Gamma_1,Gamma_2]


        if FirstPiolaKirchhoffStress_Compressible == 1:
            P = FirstPiolaKirchhoffStress_Compressible_Mooney_Rivlin(c10,c01,u,p)
        if FirstPiolaKirchhoffStress_Compressible == 2:
            P = FirstPiolaKirchhoffStress_Compressible_Gent(E,Jm,u,p)
        if FirstPiolaKirchhoffStress_Compressible == 3:
            P = FirstPiolaKirchhoffStress_Compressible_Haines_Wilson(c10,c01,c20,c02,c30,c11,u,p)

        F = DeformationGradient(u)
        J = det(F)
        C = F.T*F


        # construction of the bilinear form
        n = FacetNormal(mesh)
        h = CellDiameter(mesh)

        # Define variational problem
        A = inner(P,grad(v))*dx - (det(C)-1.0)*q*dx#+ 0.001*inner(grad(u),grad(v))*dx
        dA = derivative(A,up,dup)
        L = dot(B,v)*dx

        # Compute solution
        problem = NonlinearVariationalProblem(A-L, up, bcs,dA)
        solver = NonlinearVariationalSolver(problem)
        prm = solver.parameters
        prm['newton_solver']['absolute_tolerance'] = 1E-7
        prm['newton_solver']['relative_tolerance'] = 1E-8
        prm['newton_solver']['maximum_iterations'] = 50
        #prm['newton_solver']['relaxation_parameter'] = 1.0
        prm["newton_solver"]['linear_solver']="mumps"
        solver.solve()
        (u,p) = split(up)

        
    # quantity of interest
    if interest == 1 or interest==5:
        if FirstPiolaKirchhoffStress_Compressible == 1:
            P = FirstPiolaKirchhoffStress_Compressible_Mooney_Rivlin(c10,c01,u,p)
        if FirstPiolaKirchhoffStress_Compressible == 2:
            P = FirstPiolaKirchhoffStress_Compressible_Gent(E,Jm,u,p)
        if FirstPiolaKirchhoffStress_Compressible == 3:
            P = FirstPiolaKirchhoffStress_Compressible_Haines_Wilson(c10,c01,c20,c02,c30,c11,u,p)
    if interest == 1:
        Q=assemble(inner(dot(P,n),n)*ds(NEUMANN_BOUNDARY))
    if interest == 2:
        Q=assemble(inner(u,u)*dx(1))
    if interest == 3:
        Q=assemble(inner(grad(u),grad(u))*dx(1))
    if interest == 4:
        Q=assemble(tr(GreenLagrange(u))*dx(1))
    if interest == 5:
        Q=assemble(VonMisesStress(P)*dx(1))

    print("quantity of interrest : ", Q )
    Q_array[ite] = Q
    
    
    # resolution of the dual system
    if uniform == False and ite < max_ite-1:        
        if (estim == 1 or estim == 2 or estim == 4):
	        # Define dual variational problem in Pk
	        # Define functions
            vq_h = TestFunction(VQ)
            (v_h, q_h)  = TestFunctions(VQ)           # Test function
            zp_h = Function(VQ)
            (z_h,p_h) = split(zp_h)
            dzp_h = TrialFunction(VQ)    
            (dz_h,dp_h) = split(dzp_h)
            A = inner(P,grad(dz_h))*dx - (det(C)-1.0)*dp_h*dx 
            dA = derivative(A,up,vq_h)
            if interest ==1:
                Q = inner(dot(P,n),n)*ds(NEUMANN_BOUNDARY) 
            if interest ==2:
                Q = inner(u,u)*dx(1)  
            if interest ==3:
                Q = inner(grad(u),grad(u))*dx(1)  
            if interest == 4:
                Q=tr(GreenLagrange(u))*dx(1)
            if interest == 5:
                Q=VonMisesStress(P)*dx(1)
            dQ = derivative(Q,up,vq_h)

            # Dirichlet boundary conditions
            Gamma_1 = DirichletBC(VQ.sub(0), Constant((0.0,0.0,0.0)), boundaries, DIRICHLET_BOUNDARY)
            Gamma_2 = DirichletBC(VQ.sub(0), Constant((0.0,0.0,0.0)), boundaries, NEUMANN_BOUNDARY)
            bcs = [Gamma_1,Gamma_2]

            # Compute solution
            problem = LinearVariationalProblem(dA,dQ, zp_h, bcs)
            solver = LinearVariationalSolver(problem)
            prm = solver.parameters
            prm['linear_solver'] = "mumps"
            solver.solve()
            (z_h,p_h) = split(zp_h)

        if (estim == 3 or estim == 4):
	        # Define dual variational problem in P(k+1)
            vq = TestFunction(VQ2)
            (v, q)  = TestFunctions(VQ2)           # Test function
            zp = Function(VQ2)
            (z,p_dual) = split(zp)
            dzp = TrialFunction(VQ2)    
            (dz,dp) = split(dzp)
            A = inner(P,grad(dz))*dx - (det(C)-1.0)*dp*dx #+ 0.001*inner(grad(u),grad(v))*dx
            dA = derivative(A,up,vq)
            if interest == 1:
                Q = inner(dot(P,n),n)*ds(NEUMANN_BOUNDARY)
            if interest == 2:
                Q = inner(u,u)*dx(1)
            if interest ==3:
                Q = inner(grad(u),grad(u))*dx(1)
            if interest == 4:
                Q=tr(GreenLagrange(u))*dx(1)    
            if interest == 5:
                Q=VonMisesStress(P)*dx(1) 
            dQ = derivative(Q,up,vq)

            # Dirichlet boundary conditions
            Gamma_1 = DirichletBC(VQ2.sub(0), Constant((0.0,0.0,0.0)), boundaries, DIRICHLET_BOUNDARY)
            Gamma_2 = DirichletBC(VQ2.sub(0), Constant((0.0,0.0,0.0)), boundaries, NEUMANN_BOUNDARY)
            bcs = [Gamma_1,Gamma_2]

            # Compute solution
            problem = LinearVariationalProblem(dA,dQ, zp, bcs)
            solver = LinearVariationalSolver(problem)
            prm = solver.parameters
            prm['linear_solver'] = "mumps"
            solver.solve()
            (z,p_dual) = split(zp)

        # Computation of the estimator eta_H
        if (estim == 1 or estim == 2 or estim == 4):
	        Ezp_h = Function(VQ2)
	        Ezp_h.extrapolate(zp_h)
        if estim == 1:
	        dif = Ezp_h - zp_h
        if estim == 2:
	        dif = Ezp_h - interpolate(Ezp_h,VQ)
        if estim == 3:
	        dif = zp - interpolate(zp,VQ)
        if estim == 4:
	        dif = zp - zp_h
        dif = project(dif,VQ2)
        diffz, diffp = split(dif)
        Ez_h, Ep_h = split(Ezp_h)
        if (estim == 1 or estim == 2 or estim == 4):
	        rEz_h = assemble(inner(P,grad(Ez_h))*dx - (det(C)-1.0)*Ep_h*dx - inner(dot(P,n),Ez_h)*ds)
	        r_z = assemble(inner(P,grad(z_h))*dx - (det(C)-1.0)*p_h*dx)# - inner(dot(P,n),z_h)*ds)
        else:
	        rEz_h = assemble(inner(P,grad(z))*dx - (det(C)-1.0)*p_dual*dx - inner(dot(P,n),z)*ds)
	        r_z = assemble(inner(P,grad(z))*dx - (det(C)-1.0)*p_dual*dx)# - inner(dot(P,n),z_h)*ds)
        eta_h = abs(rEz_h)
        print("diff bord",assemble(inner(dot(P,n),Ez_h)*ds))
        print("||r(E(z_h))||:",eta_h)
        print("||r(z_h)||:",r_z)

        # Computation of the estimator eta_T
        w = TestFunction(DG0)
        eta_T1 = w*inner(div(P),diffz)*dx+ w*(det(C)-1.0)*diffp*dx
        eta_T2 = -2.0*avg(w)*0.5*inner((P('+'))*n('+')+(P('-'))*n('-'),avg(diffz))*dS
        eta_T3 = w*inner( -P*n,diffz)*ds#(NEUMANN_BOUNDARY) # jump at Neumann boundary
        #eta_T4 = w*inner( -P*n,diffz)*ds(NEUMANN_BOUNDARY) # jump at Neumann boundary

        eta_T = eta_T1 + eta_T2 + eta_T3 #+ eta_T4
        eta = np.absolute( assemble(eta_T).get_local() )
        sum_eta_T = sum(eta)
        print("Sum |eta_T|: ",sum_eta_T)

        # Norm of the approximate solution
        print('Norm of the approximate solution',assemble(inner(u,u)*dx)**(0.5))
        
        # Construction of the indicator for the refinement
        cell_markers = refinement(eta, mesh)

        # Update estimator arrays
        sum_eta_T_array[ite] = sum_eta_T
        eta_array[ite] = eta_h
        

        
    # Update estimator arrays
    num_cell_array[ite] = mesh.num_cells()

    #save deformed mesh
    u_h = project(u,V)
    meshTmp=Mesh(mesh)
    ALE.move(meshTmp,u_h)
    #vtkfileItDef = File(nomFic+'{}_Deforme.pvd'.format(ite))
    #vtkfileItDef << meshTmp
        
    # print number of cells
    print('number of cells',mesh.num_cells())

    # upload numbr of iteration
    ite = ite + 1



##########################
#### Dernier rafinement uniforme pour le DWR
#####################
if uniform==False:
    meshRefine = refine(mesh)
    ajust_hole(meshRefine) 
    mesh= Mesh(meshRefine)
    #vtkfileIt = File(nomFic+'mesh_{}.pvd'.format(ite))
    #vtkfileIt << mesh


    # Define boundary segments for Neumann conditions
    domain = MeshFunction("size_t", mesh, mesh.geometric_dimension(),0)
    omega = Omega()
    omega.mark(domain,1)
    
    boundaries = MeshFunction("size_t", mesh, mesh.geometric_dimension()-1)
    boundaries.set_all(NEUMANN_BOUNDARY_NO_LOAD)

    neumann_boundary = NeumannBoundary()
    neumann_boundary.mark(boundaries, NEUMANN_BOUNDARY)

    dirichlet_boundary = DirichletBoundary()
    dirichlet_boundary.mark(boundaries, DIRICHLET_BOUNDARY)


    # normal and size of the cells
    n = FacetNormal(mesh) 
    h = CellDiameter(mesh)
    print("Number of cells :",mesh.num_cells())	

    # Construction of the different spaces
    DG0 = FunctionSpace(mesh,'DG',0)
    P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = P2 * P1
    V = FunctionSpace(mesh, P2)
    VQ = FunctionSpace(mesh, TH)
    P3 = VectorElement("Lagrange", mesh.ufl_cell(), 3)
    P2 = FiniteElement("Lagrange", mesh.ufl_cell(), 2)
    TH2 = P3 * P2
    VQ2 = FunctionSpace(mesh, TH2)

    # Initialize cell function for domains
    dx = Measure("dx")(domain=mesh,subdomain_data = domain, metadata={'quadrature_degree': 5})
    ds = Measure("ds")(domain=mesh, subdomain_data = boundaries) # exterior facet
    dS = Measure("dS")()

    
    # Define functions
    dup = TrialFunction(VQ)            # Incremental displacement
    (v, q)  = TestFunctions(VQ)             # Test function
    up  = Function(VQ)                 # Displacement from previous iteration
    (u,p) = split(up)

    displacement_array = np.linspace(0.0,57.3,nb_iter_chargement)
    max_it=len(displacement_array)
    force_array = np.zeros(max_it)
    elongation_array = np.zeros(max_it)


    for rit in range(max_it):
        print('###########################################---------iteration: ', rit)
        displacement=displacement_array[rit]
        print('displacement',displacement)

        # Dirichlet boundary conditions
        Gamma_1 = DirichletBC(VQ.sub(0), Constant((0.0,0.0,0.0)), boundaries, DIRICHLET_BOUNDARY)
        Gamma_2 = DirichletBC(VQ.sub(0), Constant((0.0,displacement,0.0)), boundaries, NEUMANN_BOUNDARY)
        bcs = [Gamma_1,Gamma_2]


        if FirstPiolaKirchhoffStress_Compressible == 1:
            P = FirstPiolaKirchhoffStress_Compressible_Mooney_Rivlin(c10,c01,u,p)
        if FirstPiolaKirchhoffStress_Compressible == 2:
            P = FirstPiolaKirchhoffStress_Compressible_Gent(E,Jm,u,p)
        if FirstPiolaKirchhoffStress_Compressible == 3:
            P = FirstPiolaKirchhoffStress_Compressible_Haines_Wilson(c10,c01,c20,c02,c30,c11,u,p)

        F = DeformationGradient(u)
        J = det(F)
        C = F.T*F


        # construction of the bilinear form
        n = FacetNormal(mesh)
        h = CellDiameter(mesh)

        # Define variational problem
        A = inner(P,grad(v))*dx - (det(C)-1.0)*q*dx#+ 0.001*inner(grad(u),grad(v))*dx
        dA = derivative(A,up,dup)
        L = dot(B,v)*dx

        # Compute solution
        problem = NonlinearVariationalProblem(A-L, up, bcs,dA)
        solver = NonlinearVariationalSolver(problem)
        prm = solver.parameters
        prm['newton_solver']['absolute_tolerance'] = 1E-7
        prm['newton_solver']['relative_tolerance'] = 1E-8
        prm['newton_solver']['maximum_iterations'] = 50
        prm["newton_solver"]['linear_solver']="mumps"
        solver.solve()
        (u,p) = split(up)
        
        if interest ==1 or interest ==5:     
            if FirstPiolaKirchhoffStress_Compressible == 1:
                P = FirstPiolaKirchhoffStress_Compressible_Mooney_Rivlin(c10,c01,u,p)
            if FirstPiolaKirchhoffStress_Compressible == 2:
                P = FirstPiolaKirchhoffStress_Compressible_Gent(E,Jm,u,p)
            if FirstPiolaKirchhoffStress_Compressible == 3:
                P = FirstPiolaKirchhoffStress_Compressible_Haines_Wilson(c10,c01,c20,c02,c30,c11,u,p)
        if interest == 1:
            Q_fine = assemble(inner(dot(P,n),n)*ds(NEUMANN_BOUNDARY))
        if interest ==2:
            Q_fine = assemble(inner(u,u)*dx(1))
        if interest ==3:
            Q_fine = assemble(inner(grad(u),grad(u))*dx(1))
        if interest == 4:
            Q_fine=assemble(tr(GreenLagrange(u))*dx(1))
        if interest == 5:
            Q_fine = assemble(VonMisesStress(P)*dx(1))

        
# computation of the model and dsicretisation erro
if interest ==1:  
    model_error = abs(Q_fine-20.0)/20.0
    print("Error model",model_error)
    for ite in range(max_ite):
        model_discretisation_error_array[ite] = abs(20.0-Q_array[ite])/20.0
for ite in range(max_ite):
    discretisation_error_array1[ite]  = abs(Q_fine-Q_array[ite])/Q_fine
    discretisation_error_array2[ite]  = abs(Q_fine-Q_array[ite])/20.0
    eta_h_effi_array[ite] = eta_array[ite]/abs(Q_fine-Q_array[ite])
    sum_eta_T_effi_array[ite] = sum_eta_T_array[ite]/abs(Q_fine-Q_array[ite])


# Function used to write in the outputs files
def output_latex(f,A,B):
	for i in range(len(A)):
		f.write('(')
		f.write(str(A[i]))
		f.write(',')
		f.write(str(B[i]))
		f.write(')\n')
	f.write('\n')




"""# Comutation of the last error map
if dual==True :
    error_map = MeshFunction("double", mesh, 2)
    for i in range(len(eta)):
        error_map[i] = abs(eta)[i]

    error_map = Function(DG0)
    for c in range(mesh.num_cells()):
    	error_map.vector()[c]=abs(eta)[c]
    print('max eta',max(abs(eta)))
"""



# save fianl boundary
File_mesh = File(nomFic+'final_mesh.xml')
File_mesh << mesh
File_mesh = File(nomFic+'final_mesh.pvd')
File_mesh << mesh
File = File(nomFic+'final_boundaries.xml')
File << boundaries

"""#save deformed mesh
u_h = project(u,V)
meshTmp=Mesh(mesh)
ALE.move(meshTmp,u_h)
vtkfileItDef = File(nomFic+'_Deforme.pvd')
vtkfileItDef << meshTmp"""


# prefixe of the file
nomFic = 'output_silicone/'
if uniform == True:
    nomFic += 'uniform'
else:
    nomFic += 'adaptative'
nomFic += '_interest_{name}'.format(name=interest)
if FirstPiolaKirchhoffStress_Compressible == 1:
	nomFic += '_mooney'
if FirstPiolaKirchhoffStress_Compressible == 2:
    nomFic += '_gent'
if FirstPiolaKirchhoffStress_Compressible == 3:
    nomFic += '_wilson'
nomFic += '_ite_{name}'.format(name=max_ite)	


# print the different arrays in the outputs files
f = open(nomFic+'.txt','w')
if uniform == True:
	f.write('Refinement : uniform \n')
else:
    f.write('Refinement : DWR \n')
f.write('Refinement Fraction : ')
f.write(str(refinement_fraction))
f.write('\n')
f.write('Degree of polynom for CG : ')
f.write(str(polV))
f.write('\n')
f.write('Quantity of interest : ')
if interest==1:
	f.write('force bord superieur (mesh with 5 holes)')
if interest==2:
    f.write('L2 Norm in the rectangle [lx/4,3lx/4]X[ly/4,3ly/4] (mesh without hole)')
if interest==3.:
    f.write('H1 Norm in the rectangle [lx/4,3lx/4]X[ly/4,3ly/4] (mesh without hole)')
if interest==4:
    f.write('trace of the green lagrange tensor (mesh without hole)')
if interest==5:
    f.write('vonmisestress (mesh without hole)')
f.write('\n')
f.write('Q(u_fine) : ')
f.write(str(Q_fine))
f.write('\n')
f.write('Q(u) : \n')	
output_latex(f,num_cell_array,Q_array)
f.write('Eta_h : \n')	
output_latex(f,num_cell_array,eta_array)
f.write('Sum eta_T : \n')	
output_latex(f,num_cell_array,sum_eta_T_array)
f.write('Eta_h / |J(u_h)-J(u)| : \n')	
output_latex(f,num_cell_array,eta_h_effi_array)
f.write('Sum eta_T / |J(u_h)-J(u)| : \n')	
output_latex(f,num_cell_array,sum_eta_T_effi_array)
output_latex(f,num_cell_array,sum_eta_T_effi_array)
f.write('Sum eta_T / |J(u_h)-J(u)| : \n')	
output_latex(f,num_cell_array,sum_eta_T_effi_array)
if interest == 1:
    f.write('relative  model error:')
    f.write(str(model_error))	
    f.write('\n')	
f.write('relative discretisation error: Q(u_h)-Q(u_fine)/Q(u_fine)\n')
output_latex(f,num_cell_array,discretisation_error_array1)
f.write('relative discretisation error: Q(u_h)-Q(u_fine)/Q(u_exact)\n')
output_latex(f,num_cell_array,discretisation_error_array2)
f.write('relative discretisation and model error:\n')
output_latex(f,num_cell_array,model_discretisation_error_array)
f.close()












































