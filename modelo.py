import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import pickle

# Função para carregar imagens
def load_images_from_folders(folder_path, label):
    data = []
    labels = []
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"O diretório especificado não foi encontrado: {folder_path}")
    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".png", ".jpeg", ".webp")):
            img_path = os.path.join(folder_path, file)
            image = load_img(img_path, target_size=(64, 64), color_mode='grayscale')
            image = img_to_array(image).flatten()
            data.append(image)
            labels.append(label)
    return data, labels

# Caminhos das pastas
calvos_path = "C:/Users/Luquinha/Desktop/godigos/CalvicieModelo/Calvo"
cabeludos_path = "C:/Users/Luquinha/Desktop/godigos/CalvicieModelo/Cabeludo"

# Carregar dados
calvos_data, calvos_labels = load_images_from_folders(calvos_path, "calvo")
cabeludos_data, cabeludos_labels = load_images_from_folders(cabeludos_path, "cabeludo")

# Combinar os dados e rótulos
data = np.array(calvos_data + cabeludos_data)
labels = np.array(calvos_labels + cabeludos_labels)

# Verificar distribuição do dataset
unique, counts = np.unique(labels, return_counts=True)
print("Distribuição no dataset completo:", dict(zip(unique, counts)))

# Codificar os rótulos
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

# Dividir o dataset (estratificado)
X_train, X_test, y_train, y_test = train_test_split(
    data, encoded_labels, test_size=0.5, stratify=encoded_labels
)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Aplicar Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Transformar as imagens para Data Augmentation
X_train_augmented = []
y_train_augmented = []

# Adicionar variações às imagens do treinamento
for i, image in enumerate(X_train):
    image = image.reshape(64, 64, 1)  # Reshape para 2D (necessário para o ImageDataGenerator)
    augmented_images = datagen.flow(np.expand_dims(image, axis=0), batch_size=1)
    for _ in range(5):  # Criar 5 variações de cada imagem
        augmented_image = next(augmented_images)  # Corrigido para usar `next`
        X_train_augmented.append(augmented_image[0].flatten())  # Pega a imagem transformada
        y_train_augmented.append(y_train[i])


X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

# Criar o modelo ajustado com mais camadas e variabilidade
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),  # Mais camadas ocultas para maior complexidade
    activation='relu',
    solver='adam',
    alpha=0.05,  # Regularização para evitar overfitting
    learning_rate_init=0.00005,
    max_iter=30,  # Número de épocas
    tol=0,  # Sem parada antecipada
    verbose=False  # Logs desativados para melhor visualização
)

# Variáveis para armazenar os resultados durante o treinamento
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# Treinar o modelo manualmente por época para capturar as métricas
for epoch in range(1, mlp.max_iter + 1):
    mlp.partial_fit(X_train_augmented, y_train_augmented, classes=np.unique(y_train_augmented))
    
    # Calcular perda e acurácia no treinamento
    train_loss = log_loss(y_train_augmented, mlp.predict_proba(X_train_augmented))
    train_acc = accuracy_score(y_train_augmented, mlp.predict(X_train_augmented))
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    # Calcular perda e acurácia no teste
    test_loss = log_loss(y_test, mlp.predict_proba(X_test))
    test_acc = accuracy_score(y_test, mlp.predict(X_test))
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

# Avaliar o modelo no conjunto de teste
y_pred_test = mlp.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)

# Exibir os resultados
print("\nAcurácia no conjunto de teste:", accuracy_test)
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred_test, target_names=encoder.classes_))

# Diretório de destino
output_directory = "C:/Users/Luquinha/Desktop/godigos/CalvicieModelo"

# Salvar o modelo e o scaler
with open("modeloCalvos.pkl", "wb") as model_file:
    pickle.dump(mlp, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

# Plotar gráficos de perda e acurácia
plt.figure(figsize=(14, 6))

# Gráfico de perda
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Perda - Treinamento', color='blue')
plt.plot(test_losses, label='Perda - Teste', color='orange')
plt.title('Curva de Perda')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

# Gráfico de acurácia
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Acurácia - Treinamento', color='blue')
plt.plot(test_accuracies, label='Acurácia - Teste', color='orange')
plt.title('Curva de Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

plt.tight_layout()
plt.show()
