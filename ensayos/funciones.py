from scipy.signal import butter, lfilter,convolve, detrend, welch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def model_opensees():
    import openseespy.opensees as ops
    import opsvis as opsv
    import matplotlib.pyplot as plt
    import numpy as np

    # propiedades geometricas de la viga
    bv = 1*2.54 #cm
    hv = (1/4)*2.54 #cm

    # propiedades geometricas de la columna
    bc = 1*2.54 #cm
    hc = (1/8)*2.54 #cm

    # propiedades mecanicas
    E = 2039432.43 #kgf/cm2
    E = 1.3815*E
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

    # Obtenemos los periodos y las frecuencias
    Nmodes = 4
    λ = ops.eigen(Nmodes)
    Tn,f=[],[]
    for i in range(len(λ)):
        Tn.append(2.0*np.pi/λ[i]**0.5)
        f.append(1/Tn[i])
    f=np.array(f)
    
    #obtenemos los amortiguamientos
    xi = 0.02
    c=np.array([xi,xi,xi,xi])

    # Obtenemos las formas modales
    modos = []
    for i in range(4):
        for j in range(4):
            modos.append(ops.nodeEigenvector(2*(j+2),i+1,1))
    modos = np.array(modos)
    modos=np.reshape(modos,(4,4))
    modos = modos / np.abs(modos).max(axis=1).reshape(-1, 1)
    
    return f, c, modos
    
def leer_data(lista,col):
    factor=9.81/16384.0
    #factor=1
    sensor1 = np.genfromtxt(lista[0], delimiter=',',usecols=col)
    sensor2 = np.genfromtxt(lista[1], delimiter=',',usecols=col)
    sensor3 = np.genfromtxt(lista[2], delimiter=',',usecols=col)
    sensor4 = np.genfromtxt(lista[3], delimiter=',',usecols=col)
    data = np.vstack((factor*sensor1, factor*sensor2, factor*sensor3, factor*sensor4))
    return data

def plot_pwelch(matrix, fs):
    num_vectors = matrix.shape[0]  # Número de vectores en la matriz
    fig, axs = plt.subplots(num_vectors, 1, figsize=(8, 6*num_vectors))
    
    for i in range(num_vectors):
        vector = matrix[i, :]  # Obtener el vector actual
        f, Pxx = welch(vector, fs=fs)  # Calcular el PSD utilizando welch
        
        axs[i].plot(f, Pxx)  # Graficar PSD en escala logarítmica
        axs[i].set_xlabel('Frecuencia [Hz]')
        axs[i].set_ylabel('Densidad espectral de potencia [V^2/Hz]')
        axs[i].set_title(f'PSD del vector {i+1}')
        
    plt.tight_layout()
    plt.show()
    
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
	"""
	Función que hace un filtrado de frecuencias en pasabanda por el método Butterworth.
            
	PARÁMETROS:
	data : narray de la señal a filtrar.
	Lowcut : Límite inferior del pasabanda de frecuencias (Hz).
	higcut : Límite superior del pasabanda de frecuencias (Hz).
	fs : Frecuencia de muestreo de la data (Hz).
	order: Orden del Butterworth.

	RETORNOS:
	y : narray de la señal filtrada.
	"""
	nyq = 0.5*fs
	low = lowcut / nyq
	high = highcut / nyq

	b, a = butter(order, [low, high], btype='bandpass', analog=False)
	y = lfilter(b, a, data)

	return y

def procesamiento(data,fs,fmin,fmax):
  ax=np.array(data)
  Np=ax.shape[0]
  n=len(ax[0])
  dt=1/fs
  t=[x*dt for x in range(n)]
  for i in range(Np):
    ax[i]=detrend(ax[i])
 
  ##############CORRECCION POR PASABANDA###############
  for j in range(Np):
    ax[j]=butter_bandpass_filter(ax[j],fmin,fmax,fs)
  
  # PARA EL GRAFICO
  a=19
  b=5
  plt.figure( figsize = (a,b))
  for q in range(Np):
    plt.plot(t, ax[q],alpha=0.7)
  plt.title("ACELERACION-TIEMPO")
  plt.grid(True)
  plt.show()

  # Devuelve la lista con los vectores de datos
  return t, ax

def signals_processing(datos,fs,fmin,fmax):
    axi = datos
    t, axf= procesamiento(axi,fs,fmin,fmax)
    plot_pwelch(axi, fs)
    plot_pwelch(axf, fs)
    return axf

def modal_shape_plot(fn,ms,tittle):
    colors = ["b", "g", "r", "c", "m", "y", "k", "b", "g", "r", "c", "m", "y", "k", "b", "g", "r", "c", "m", "y", "k"]
    mss = np.insert(ms, 0, 0, axis=1)
    fig, ax = plt.subplots(nrows=1, ncols=len(fn), figsize=(20, 5))
    for i in range(len(fn)):
        ax[i].plot(np.concatenate(([mss[i][0]], [mss[i][1]], [mss[i][2]], [mss[i][3]], [mss[i][4]])),[0, 1, 2, 3, 4], linestyle='--', marker='o', color=colors[i])
        ax[i].set_title("M{}-f: {:.4f} Hz".format(i + 1, np.round(fn[i],4)))
    fig.suptitle(tittle)
    #ax[0].legend()
    plt.show()

def mac_value(ms0, ms1):
  mac = np.abs(ms0.T @ ms1) ** 2 / (ms0.T @ ms0) / (ms1.T @ ms1)
  if mac > 1.0:
      mac = 1.0
  return mac

def comparacion(f1,f2,zeta1,zeta2,ms1,ms2,l):

  col=[]
  for i in range(l):
    col.append(f"modo {i+1}")

  ##----comparacion de frecuencias----------##
  datosf = {
      'Modos vib.': col,
      'Numerico(Hz)': f1,
      'Experimental(Hz)': f2,
      'Error relativo(%)': 100*(np.abs(f1-f2)/f1)
  }
  df = pd.DataFrame(datosf)
  df = df.round(6)
  df = df.to_string(index=False)
  ##----comparacion de factor de amortiguamiento----------##
  datosz = {
      'Modos vib.': col,
      'Numerico(%)': 100*zeta1,
      'Experimental(%)': 100*zeta2,
      'Error relativo(%)': 100*(np.abs(zeta1-zeta2)/zeta1)
  }
  dz = pd.DataFrame(datosz)
  dz = dz.round(6)
  dz = dz.to_string(index=False)
  ##------------comparacion de formas modales-------------##
  mac=[]
  for i in range(l):
    for j in range(l):
      mac.append(mac_value(ms1[i], ms2[j]))
  mac=np.array(mac)
  mac=np.reshape(mac,(l,l))
  MAC = pd.DataFrame(mac, columns=col, index=col)

  fig, ax = plt.subplots()
  sns.heatmap(MAC,cmap="viridis",ax=ax,annot=True, fmt='.4f',)
  fig.tight_layout()
  plt.show()
  
  np.set_printoptions(linewidth=1000, precision=3)
  np.set_printoptions(linewidth=1000,formatter={"float": lambda x:"%10.5f"%(x)})
  print("COMPARACION DE FRECUENCIAS")
  print(df)
  print(" ")
  print("COMPARACION DE AMORTIGUAMIENTOS")
  print(dz)
  print(" ")
  print("FORMAS MODALES PREDICHAS NUMERICAMENTE")
  print(ms1)
  print(" ")
  print("FORMAS MODALES EXPERIMENTALES")
  print(ms2)
  print(" ")
  print("COMPARACION DE FORMAS MODALES")
  print(mac)
  print(" ")
  return df, dz, mac

