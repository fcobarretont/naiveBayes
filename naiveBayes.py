import os
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Função para carregar imagens e extrair características HOG
def load_data(data_dir):
    X = []
    y = []
    
    # Definir o tamanho fixo para redimensionamento
    image_size = (128, 128)  # Tamanho desejado (altura, largura)
    
    # Carregar imagens de gatos
    for img_name in os.listdir(os.path.join(data_dir, 'gatos')):
        img_path = os.path.join(data_dir, 'gatos', img_name)
        image = imread(img_path)
        image = resize(image, image_size)  # Redimensionar a imagem
        
        if image.shape[2] == 4:
            image = image[:, :, :3]  # Remove o canal alfa
        
        gray_image = rgb2gray(image)
        features = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        X.append(features)
        y.append('gato')

    # Carregar imagens de não gatos (ex: cachorros)
    for img_name in os.listdir(os.path.join(data_dir, 'nao_gatos')):
        img_path = os.path.join(data_dir, 'nao_gatos', img_name)
        image = imread(img_path)
        image = resize(image, image_size)  # Redimensionar a imagem
        
        if image.shape[2] == 4:
            image = image[:, :, :3]  # Remove o canal alfa
        
        gray_image = rgb2gray(image)
        features = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        X.append(features)
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

# Função para classificar uma nova imagem
def classify_image(image_path):
    # Carregar a imagem
    image = imread(image_path)

    # Redimensionar a imagem antes de processá-la
    image_size = (128, 128)  # Deve ser o mesmo tamanho usado no treinamento
    image_resized = resize(image, image_size)  # Redimensionar a imagem
    
    if image_resized.shape[2] == 4:
        image_resized = image_resized[:, :, :3]  # Remove o canal alfa
    
    gray_image = rgb2gray(image_resized)
    features = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)

    prediction = model.predict([features])[0]
    
    display_result(image_resized, prediction)

def display_result(image_array, prediction):
    """Exibe a imagem e o resultado na interface."""
    
    # Converter a imagem do formato NumPy para PIL para exibição no tkinter
    image_pil = Image.fromarray((image_array * 255).astype(np.uint8))  # Convertendo para uint8
    
    # Atualizar label com a imagem
    img_tk = ImageTk.PhotoImage(image_pil)
    label_image.config(image=img_tk)
    label_image.image = img_tk  # Manter uma referência à imagem
    
    # Atualizar label com o resultado da classificação
    label_result.config(text=f'Classificação: {prediction}')

# Função chamada ao clicar no botão "Selecionar Imagem"
def select_image():
    file_path = filedialog.askopenfilename(title="Selecione uma imagem", 
                                            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    
    if file_path:
        classify_image(file_path)

# Criar interface gráfica com tkinter
root = tk.Tk()
root.title("Classificador de Gatos")
root.geometry("600x400")  # Aumenta o tamanho da janela

# Criar botão para selecionar imagem
btn_select_image = tk.Button(root, text="Selecionar Imagem", command=select_image)
btn_select_image.pack(pady=20)

# Label para exibir a imagem classificada
label_image = tk.Label(root)
label_image.pack(pady=10)

# Label para exibir o resultado da classificação
label_result = tk.Label(root, text="", font=("Helvetica", 16))
label_result.pack(pady=10)

# Iniciar o loop da interface gráfica
root.mainloop()