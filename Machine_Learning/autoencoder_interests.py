from keras.models import Model
from keras.layers import Input, Dense
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Normalise
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Fixed dimensions
input_dim = data.shape[1] # 8
encoding_dim = 3
# Number of neurons in each Layer [8, 6, 4, 3, ...] of encoders
input_layer = Input(shape=(input_dim, ))
encoder_layer_1 = Dense(6, activation="tanh", activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder_layer_2 = Dense(4, activation="tanh")(encoder_layer_1)
encoder_layer_3 = Dense(encoding_dim, activation="tanh")(encoder_layer_2)

# Crear encoder model
encoder = Model(inputs=input_layer, outputs=encoder_layer_3)

# Use the model to predict the factors which sum up the information of interest rates.
encoded_data = pd.DataFrame(encoder.predict(data_scaled))
encoded_data.columns = ['factor_1', 'factor_2', 'factor_3']