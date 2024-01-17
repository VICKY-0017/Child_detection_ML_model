import gradio as gr
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
from PIL import Image

# Assume the input shape is (128, 128, 1) for grayscale images
input_shape = (128, 128, 1)

inputs = Input(shape=input_shape)
conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape)(inputs)
maxp_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(maxp_1)
maxp_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
conv_3 = Conv2D(128, kernel_size=(3, 3), activation='relu')(maxp_2)
maxp_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu')(maxp_3)
maxp_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)
flatten = Flatten()(maxp_4)
dense_1 = Dense(256, activation='relu')(flatten)
dense_2 = Dense(256, activation='relu')(flatten)
dropout_1 = Dropout(0.3)(dense_1)
dropout_2 = Dropout(0.3)(dense_2)
output_1 = Dense(1, activation='sigmoid', name='gender_out')(dropout_1)
output_2 = Dense(1, activation='relu', name='age_out')(dropout_2)

model = Model(inputs=[inputs], outputs=[output_1, output_2])
model.compile(loss=['binary_crossentropy', 'mae'], optimizer='adam', metrics=['accuracy'])

# Load the model weights
model.load_weights('model_weights.h5', by_name=True)

# Define a function for making predictions
def predict_age_gender(image):
    try:
        # Preprocess the input image
        image = Image.fromarray((image * 255).astype('uint8'))  # Convert image to PIL Image
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((128, 128), resample=Image.LANCZOS)  # Resize using LANCZOS
        image = np.array(image).reshape(1, 128, 128, 1) / 255.0  # Reshape and normalize (grayscale image)
        
        # Make predictions using the model
        pred = model.predict(image)
        
        # Determine age group
        predicted_age_group = "Child" if pred[1][0] <= 12 else "Not a Child"
        
        return f"Age Group: {predicted_age_group}"
    except Exception as e:
        return f"Error: {str(e)}"

# Create a Gradio interface
iface = gr.Interface(fn=predict_age_gender, inputs="image", outputs="text", live=True)

# Launch the interface
iface.launch(share  = True)
