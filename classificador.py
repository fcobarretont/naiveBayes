import os
import pickle
from flask import Flask, request, render_template
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize

app = Flask(__name__)

# Carregar o modelo treinado
with open('modelo_naive_bayes.pkl', 'rb') as file:
    model = pickle.load(file)

def classify_image(image_path):
    # Carregar a imagem e processá-la para classificação
    image = imread(image_path)
    image_size = (128, 128)  
    image_resized = resize(image, image_size)  
    
    if image_resized.shape[2] == 4:
        image_resized = image_resized[:, :, :3]  
    
    gray_image = rgb2gray(image_resized)
    features = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)

    prediction = model.predict([features])[0]
    
    return prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_file = None
    
    if request.method == 'POST':
        # Obter a imagem enviada pelo usuário
        image_file = request.files['image']
        
        if image_file:
            # Salvar a imagem na pasta 'static'
            image_path = os.path.join('static', image_file.filename)
            image_file.save(image_path)

            # Classificar a imagem usando o modelo treinado
            prediction = classify_image(image_path)

            # Não é necessário excluir a imagem após a classificação agora

    return render_template('index.html', prediction=prediction, image_file=image_file.filename if image_file else None)

if __name__ == '__main__':
    app.run(debug=True)