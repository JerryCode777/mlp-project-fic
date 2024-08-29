import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops
import opsvis as opsv
import time


####modelo en opensees######
# propiedades geometricas de la viga
bv = 1*2.54 #cm
hv = (1/4)*2.54 #cm

# propiedades geometricas de la columna
bc = 1*2.54 #cm
hc = (1/8)*2.54 #cm

# propiedades mecanicas
E = 2039432.43 #kgf/cm2
E = 1.385*E
Avig = bv*hv #cm2
Acol = bc*hc #cm2
Ivig = (bv*hv**3)/12 #cm4
Icol = (bc*hc**3)/12 #cm4
g = 981 #cm/s2


# definir el tipo de modelo a calcular
ops.wipe()
ops.model('basic', '-ndm', 2, '-ndf', 3)

# definir los nudos del problema
ops.node(1,0.,0.)
ops.node(2,50,0.)
ops.node(3,0.,30)
ops.node(4,50,30)
ops.node(5,0,60)
ops.node(6,50,60)
ops.node(7,0.,90)
ops.node(8,50,90)
ops.node(9,0.,120)
ops.node(10,50,120)

# definir las masas nodales para determinar la matriz de masa
ops.mass(3,1.247/g,0.,0.)
ops.mass(4,1.247/g,0.,0.)
ops.mass(5,1.247/g,0.,0.)
ops.mass(6,1.247/g,0.,0.)
ops.mass(7,1.247/g,0.,0.)
ops.mass(8,1.247/g,0.,0.)
ops.mass(9,1.151/g,0.,0.)
ops.mass(10,1.151/g,0.,0.)

# definir la transformación de coordenadas
ops.geomTransf('Linear', 1)


# definir nuestras barras para determinar la matriz de rigidez
ops.element('elasticBeamColumn', 1, 1, 3, Acol, E, Icol, 1)
ops.element('elasticBeamColumn', 2, 2, 4, Acol, E, Icol, 1)
ops.element('elasticBeamColumn', 3, 3, 4, Avig, E, Ivig, 1)
ops.element('elasticBeamColumn', 4, 3, 5, Acol, E, Icol, 1)
ops.element('elasticBeamColumn', 5, 4, 6, Acol, E, Icol, 1)
ops.element('elasticBeamColumn', 6, 5, 6, Avig, E, Ivig, 1)
ops.element('elasticBeamColumn', 7, 5, 7, Acol, E, Icol, 1)
ops.element('elasticBeamColumn', 8, 6, 8, Acol, E, Icol, 1)
ops.element('elasticBeamColumn', 9, 7, 8, Avig, E, Ivig, 1)
ops.element('elasticBeamColumn', 10, 7, 9, Acol, E, Icol, 1)
ops.element('elasticBeamColumn', 11, 8, 10, Acol, E, Icol, 1)
ops.element('elasticBeamColumn', 12, 9, 10, Avig, E, Ivig, 1)

# ingreso de restricciones
ops.fix(1, 1, 1, 1)
ops.fix(2, 1, 1, 1)

opsv.plot_model(fig_wi_he=(20., 14.))
plt.show()

Nmodes=4

start_time = time.time()
λ = ops.eigen(Nmodes)
total_time=time.time() - start_time
print("--- %s seconds ---" % total_time)

Tn=[]
for i in λ:
    Tn.append(1/(2.0*np.pi/i**0.5))
print(Tn)

for i in range(Nmodes):
    opsv.plot_mode_shape(i+1,1)
    plt.show()
ops.integrator("LoadControl", 0, 1,0,0)
ops.test("EnergyIncr",1.0e-10,100,0)
ops.algorithm("Newton")
ops.numberer("RCM")
ops.constraints("Transformation")
ops.system("ProfileSPD")
ops.analysis("Static")

print(ops.nodeEigenvector(9,1,1))
print(ops.nodeEigenvector(9,2,1))
print(ops.nodeEigenvector(9,3,1))
print(ops.nodeEigenvector(9,4,1))


aux=[]
for j in range(Nmodes):
    aux.append(ops.nodeEigenvector(9,j+1,1)*100000)
print(aux)