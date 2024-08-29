import numpy as np
import tensorflow as tf
import math as math
from tensorflow import keras
import matplotlib.pyplot as plot
import os
from keras.models import Sequential
from keras.layers import Dense


def leerdato(nombre):
    f = open (nombre,'r')
    mensaje = f.read()
    valores=mensaje.split('\n')
    filtrado=[]

    for i in valores:
        if(i!=""):
            filtrado.append(i.split())

    training_data=[]
    for i in range(len(filtrado)):
        aux=[]
        for j in range(len(filtrado[0])):
            aux.append(float(filtrado[i][j]))
        training_data.append(aux)

    f.close()
    return  training_data

target_data = leerdato('data/Answer.txt')
training_data = leerdato('data/dataSet.txt')

junto=[]
for i in range(len(training_data)):
    junto.append(training_data[i]+target_data[i])

tf.random.shuffle(junto)


n=len(training_data[0])
na=len(junto[0])
junto=np.array(junto)
s_training_data = (junto[:,0:n]).tolist()
s_target_data = (junto[:,n:na]).tolist()

#Arquitectura con mse de  1.29 % , top EPOC 200 # BUENOS RESULTADOS
def create_model27():
    model = Sequential()
    model.add(Dense(20, input_dim=len(s_training_data[0]), activation='relu'))
    model.add(Dense(50, input_dim=20, activation='relu'))
    model.add(Dense(120, input_dim=50, activation='relu'))
    model.add(Dense(100, input_dim=120, activation='relu'))
    model.add(Dense(50, input_dim=100, activation='relu'))
    model.add(Dense(len(s_target_data[0]), activation='linear', name="salida"))
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mse'])
    model.summary()
    return model


model=create_model27()
checkpoint_path = 'training_27/cp.weights.h5'
#checkpoint_path = 'training_27/cp.weights.h5'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)

hist=model.fit(np.array(s_training_data),np.array(s_target_data), epochs=1000, validation_split=0.2,callbacks=[cp_callback])

checkpoint_path = 'training_27/cp.weights.h5'
#checkpoint_path = "training_27/cp.ckpt"
model.load_weights(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)

# evaluamos el modelo
scores = model.evaluate(s_training_data, s_target_data)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print (model.predict(s_training_data).round())

history_mse=hist.history['mse']
val_mse=hist.history['val_mse']
history_loss=hist.history['loss']
val_loss=hist.history['val_loss']


ff = open ('curvaaprendisaje1.txt','w')
aux=""
for j in history_mse:
    aux=aux+str(j)+" "
aux=aux+"\n"
ff.write(aux)

aux=""
for j in val_mse:
    aux=aux+str(j)+" "
aux=aux+"\n"
ff.write(aux)

aux=""
for j in history_loss:
    aux=aux+str(j)+" "
aux=aux+"\n"
ff.write(aux)

aux=""
for j in val_loss:
    aux=aux+str(j)+" "
aux=aux+"\n"
ff.write(aux)

ff.close()

salto=1
#Para MSE
plot.plot(history_mse[salto:])
plot.plot(val_mse[salto:])
plot.title('Model accuracy')
plot.ylabel('mse')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='upper left')
plot.show()

plot.plot(history_loss[salto:])
plot.plot(val_loss[salto:])
plot.title('Model loss')
plot.ylabel('Loss')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='upper left')
plot.show()

eval_data = leerdato('ValdataSet.txt')
resp_data = leerdato('ValAnswer.txt')

# evaluamos el modelo
scores = model.evaluate(eval_data, resp_data)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print (model.predict(eval_data).round())

predictions = model.predict(eval_data)

for i in range(len(predictions)//10):
    for j in range(len(predictions[0])):
        print(round(((predictions[i][j]-resp_data[i][j])*100),2),end="%   ")
    print(" ")

interv=len(predictions)//10
print(interv)

data=[]
for i in range(len(predictions)):
    if(i%interv==0):
        fila1=[]
        for j in range(len(predictions[0])):
            print( round(predictions[i][j],1),end="%   ")
            fila1.append(round(predictions[i][j],1))
        print(" ")
        
        fila2=[]
        for j in range(len(predictions[0])):
            print(resp_data[i][j],end="%   ")
            fila2.append(resp_data[i][j])
        print(" ")
        print(" ")
        data.append([fila1,fila2])


for i in data:
    print(i)


# Crea múltiples subgrafos
fig, ax = plot.subplots(nrows=5, ncols=2, figsize=(10,20))

index=0
for i in range(5):
    for j in range(2):
        # Dibujar gráficos
        ax[i, j].plot(data[index][0],[0,1,2,3], linestyle='--', marker='o', color='b', label='Prediccion') 
        ax[i, j].set_title("Caso "+str(index+1)) 
        ax[i, j].plot(data[index][1],[0,1,2,3], linestyle='--', marker='o', color='r', label='Real') 
        index=index+1
        ax[i, j].legend()


for index in range(len(data)):
    fig, ax = plot.subplots(figsize=(5,5))
    # Dibujar gráfico
    ax.plot(data[index][0], [0,1,2,3], linestyle='--', marker='o', color='b', label='Predicción') 
    ax.plot(data[index][1], [0,1,2,3], linestyle='--', marker='o', color='r', label='Real') 
    ax.set_title("Caso "+str(index+1)) 
    ax.legend()

# Establecer título
plot.suptitle("Resultados")

# Mostrar
plot.show()
