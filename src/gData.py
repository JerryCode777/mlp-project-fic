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
    Tn.append(2.0*np.pi/i**0.5)
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

ops.remove('ele',1)
ops.remove('ele',2)
ops.remove('ele',3)
ops.remove('ele',4)
ops.remove('ele',5)
ops.remove('ele',6)
ops.remove('ele',7)
ops.remove('ele',8)
ops.remove('ele',9)
ops.remove('ele',10)
ops.remove('ele',11)
ops.remove('ele',12)


intervalos=12
cantidadtotal=pow(intervalos,4)
print(cantidadtotal)
print("Estimado en Minutos: ")
print(total_time*cantidadtotal/60)

print("###########################################################################################################################################################")


ops.wipeAnalysis()
paso=(0.9)/(intervalos)
precision=5

ff = open ('data/dataSet.txt','w')
ff2 = open ('data/Answer.txt','w')

nro=0
damage=[0]*4

for i1 in range(intervalos):
 for i2 in range(intervalos):
  print("Porcentaje: "+str(nro*100/cantidadtotal)+"  %")
  for i3 in range(intervalos):
   for i4 in range(intervalos):
          nro=nro+1
          cadena=""
          cadena2=""

          damage[0]=paso*(i1+1)
          damage[1]=paso*(i2+1)
          damage[2]=paso*(i3+1)
          damage[3]=paso*(i4+1)

          for i in range(3):
            cadena2=cadena2+str(round(damage[i],precision))+" "
          cadena2=cadena2+str(round(damage[3],precision))+"\n"
          ff2.write(cadena2)
          
          #CREACION DE ELEMENTOS
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
          ops.element('elasticBeamColumn', 1, 1, 3, Acol*damage[0], E, Icol*damage[0]**3, 1)
          ops.element('elasticBeamColumn', 2, 2, 4, Acol*damage[0], E, Icol*damage[0]**3, 1)
          ops.element('elasticBeamColumn', 3, 3, 4, Avig, E, Ivig, 1)
          ops.element('elasticBeamColumn', 4, 3, 5, Acol*damage[1], E, Icol*damage[1]**3, 1)
          ops.element('elasticBeamColumn', 5, 4, 6, Acol*damage[1], E, Icol*damage[1]**3, 1)
          ops.element('elasticBeamColumn', 6, 5, 6, Avig, E, Ivig, 1)
          ops.element('elasticBeamColumn', 7, 5, 7, Acol*damage[2], E, Icol*damage[2]**3, 1)
          ops.element('elasticBeamColumn', 8, 6, 8, Acol*damage[2], E, Icol*damage[2]**3, 1)
          ops.element('elasticBeamColumn', 9, 7, 8, Avig, E, Ivig, 1)
          ops.element('elasticBeamColumn', 10, 7, 9, Acol*damage[3], E, Icol*damage[3]**3, 1)
          ops.element('elasticBeamColumn', 11, 8, 10, Acol*damage[3], E, Icol*damage[3]**3, 1)
          ops.element('elasticBeamColumn', 12, 9, 10, Avig, E, Ivig, 1)

          # ingreso de restricciones
          ops.fix(1, 1, 1, 1)
          ops.fix(2, 1, 1, 1)

          ops.integrator("LoadControl", 0, 1,0,0)
          ops.test("EnergyIncr",1.0e-10,100,0)
          ops.algorithm("Newton")
          ops.numberer("RCM")
          ops.constraints("Transformation")
          ops.system("ProfileSPD")
          ops.analysis("Static")
          res=ops.analyze(1)
            
          lam=ops.eigen(Nmodes)
          Tmodes=np.zeros(len(lam))
          for i in range(len(Tmodes)):
           Tmodes[i]=2*np.pi/lam[i]**0.5

          #Recopilacion de los modos de vibracion en X
          #eivect=str(Tmodes[0])+" "
          eivect=""
          
          for j in range(4):
           eivect=eivect+str(round(abs(ops.nodeEigenvector(3+j*2,1,1)*1000),precision))+" "
          eivect=eivect+"\n"

          ff.write(eivect)
          ops.remove('ele',0)
          ops.remove('ele',1)
          ops.remove('ele',2)
          ops.remove('ele',3)
          ops.remove('ele',4)
          ops.remove('ele',5)
          ops.remove('ele',6)
          ops.remove('ele',7)
          ops.remove('ele',8)
          ops.remove('ele',9)
          ops.remove('ele',10)
          ops.remove('ele',11)
          ops.remove('ele',12)

          ops.wipeAnalysis()
                
ff.close()
ff2.close()
print("Termino")


intervalos=intervalos-3
cantidadtotal=pow(intervalos,4)
print(cantidadtotal)
print("Estimado en Minutos: ")
print(total_time*cantidadtotal/60)


ops.wipeAnalysis()
paso=(0.9)/(intervalos)
precision=5

ff = open ('data/ValdataSet.txt','w')
ff2 = open ('data/ValAnswer.txt','w')

nro=0
damage=[0]*4

for i1 in range(intervalos):
 for i2 in range(intervalos):
  print("Porcentaje: "+str(nro*100/cantidadtotal)+"  %")
  for i3 in range(intervalos):
   for i4 in range(intervalos):
          nro=nro+1
          cadena=""
          cadena2=""

          damage[0]=paso*(i1+1)
          damage[1]=paso*(i2+1)
          damage[2]=paso*(i3+1)
          damage[3]=paso*(i4+1)

          for i in range(3):
            cadena2=cadena2+str(round(damage[i],precision))+" "
          cadena2=cadena2+str(round(damage[3],precision))+"\n"
          ff2.write(cadena2)
          
          #CREACION DE ELEMENTOS
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
          ops.element('elasticBeamColumn', 1, 1, 3, Acol*damage[0], E, Icol*damage[0]**3, 1)
          ops.element('elasticBeamColumn', 2, 2, 4, Acol*damage[0], E, Icol*damage[0]**3, 1)
          ops.element('elasticBeamColumn', 3, 3, 4, Avig, E, Ivig, 1)
          ops.element('elasticBeamColumn', 4, 3, 5, Acol*damage[1], E, Icol*damage[1]**3, 1)
          ops.element('elasticBeamColumn', 5, 4, 6, Acol*damage[1], E, Icol*damage[1]**3, 1)
          ops.element('elasticBeamColumn', 6, 5, 6, Avig, E, Ivig, 1)
          ops.element('elasticBeamColumn', 7, 5, 7, Acol*damage[2], E, Icol*damage[2]**3, 1)
          ops.element('elasticBeamColumn', 8, 6, 8, Acol*damage[2], E, Icol*damage[2]**3, 1)
          ops.element('elasticBeamColumn', 9, 7, 8, Avig, E, Ivig, 1)
          ops.element('elasticBeamColumn', 10, 7, 9, Acol*damage[3], E, Icol*damage[3]**3, 1)
          ops.element('elasticBeamColumn', 11, 8, 10, Acol*damage[3], E, Icol*damage[3]**3, 1)
          ops.element('elasticBeamColumn', 12, 9, 10, Avig, E, Ivig, 1)

          # ingreso de restricciones
          ops.fix(1, 1, 1, 1)
          ops.fix(2, 1, 1, 1)

          ops.integrator("LoadControl", 0, 1,0,0)
          ops.test("EnergyIncr",1.0e-10,100,0)
          ops.algorithm("Newton")
          ops.numberer("RCM")
          ops.constraints("Transformation")
          ops.system("ProfileSPD")
          ops.analysis("Static")
          res=ops.analyze(1)
            
          lam=ops.eigen(Nmodes)
          Tmodes=np.zeros(len(lam))
          for i in range(len(Tmodes)):
           Tmodes[i]=2*np.pi/lam[i]**0.5

          #Recopilacion de los modos de vibracion en X
          #eivect=str(Tmodes[0])+" "
          eivect=""
          
          for j in range(4):
           eivect=eivect+str(round(abs(ops.nodeEigenvector(3+j*2,1,1)*1000),precision))+" "
          eivect=eivect+"\n"

          ff.write(eivect)
          ops.remove('ele',0)
          ops.remove('ele',1)
          ops.remove('ele',2)
          ops.remove('ele',3)
          ops.remove('ele',4)
          ops.remove('ele',5)
          ops.remove('ele',6)
          ops.remove('ele',7)
          ops.remove('ele',8)
          ops.remove('ele',9)
          ops.remove('ele',10)
          ops.remove('ele',11)
          ops.remove('ele',12)

          ops.wipeAnalysis()
                
ff.close()
ff2.close()
print("Termino")