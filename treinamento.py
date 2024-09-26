import os
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize
import pickle
from PIL import Image, ImageEnhance

# Função para carregar imagens e extrair características HOG com aumentação de dados
def load_data(data_dir):
    X = []
    y = []
    
    image_size = (128, 128)  # Tamanho desejado (altura, largura)
    
    # Carregar imagens de gatos
    for img_name in os.listdir(os.path.join(data_dir, 'gatos')):
        img_path = os.path.join(data_dir, 'gatos', img_name)
        image = imread(img_path)
        image = resize(image, image_size)  # Redimensionar a imagem

        if len(image.shape) == 3:  # checa se image tem 3 dimensoes
            if image.shape[2] == 4:
                image = image[:, :, :3]  # Remove o canal alfa
        else:
            # se img preto e branco, converta p/ rgb duplicando channels
            image = np.stack((image,) * 3, axis=-1)

        # Adicionar imagem original
        gray_image = rgb2gray(image) / 255.0  # Normaliza os valores dos pixels
        features = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        X.append(features)
        y.append('gato')

        # Aumentação: Brilho
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        enhancer = ImageEnhance.Brightness(image_pil)
        for factor in [0.5, 1.5]:  # Fatores de brilho
            bright_image = enhancer.enhance(factor)
            bright_image_resized = bright_image.resize(image_size)
            gray_image_aug = rgb2gray(np.array(bright_image_resized)) / 255.0
            features_aug = hog(gray_image_aug, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
            X.append(features_aug)
            y.append('gato')

    # Carregar imagens de não gatos (ex: cachorros)
    for img_name in os.listdir(os.path.join(data_dir, 'nao_gatos')):
        img_path = os.path.join(data_dir, 'nao_gatos', img_name)
        image = imread(img_path)
        image = resize(image, image_size)  # Redimensionar a imagem

        if len(image.shape) == 3:  # Check if the image has 3 dimensions
            if image.shape[2] == 4:
                image = image[:, :, :3]  # Remove o canal alfa
        else:
            # If the image is grayscale (2 dimensions), convert to RGB by duplicating channels
            image = np.stack((image,) * 3, axis=-1)

        # Adicionar imagem original
        gray_image = rgb2gray(image) / 255.0  # Normaliza os valores dos pixels
        features = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        X.append(features)
        y.append('nao gato')

        # Aumentação: Brilho
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        enhancer = ImageEnhance.Brightness(image_pil)
        for factor in [0.5, 1.5]:  # Fatores de brilho
            bright_image = enhancer.enhance(factor)
            bright_image_resized = bright_image.resize(image_size)
            gray_image_aug = rgb2gray(np.array(bright_image_resized)) / 255.0
            features_aug = hog(gray_image_aug, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
            X.append(features_aug)
            y.append('nao gato')

    return np.array(X), np.array(y)

# Carregar dados
data_dir = 'dataset'  # Caminho do diretório onde estão as pastas 'gatos' e 'nao_gatos'
X, y = load_data(data_dir)

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Criar e treinar o modelo Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Avaliar o modelo no conjunto de teste
y_pred = model.predict(X_test)

# Importar métricas para avaliação do modelo
from sklearn.metrics import classification_report, confusion_matrix

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# Salvar o modelo treinado
with open('modelo_naive_bayes.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Modelo treinado e salvo como 'modelo_naive_bayes.pkl'")
