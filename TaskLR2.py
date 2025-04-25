import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=1000): 
        # Инициализация весов и порога малыми случайными значениями
        self.weights = np.random.randn(input_size) * 0.01 
        self.bias = np.random.randn() * 0.01              
        self.learning_rate = learning_rate  
        self.epochs = epochs                               
        

    def activation(self, net):
        # Ступенчатая функция активации  
        return 1 if net >= 0 else 0 

    def predict(self, X):
        # Предсказание выхода для всех входных данных
        z = np.dot(X, self.weights) - self.bias
        return (z >= 0).astype(int) 

    def train(self, X, y):
        for epoch in range(self.epochs):
            weights_updated = False  

            # Итерация по всем обучающим примерам
            for i in range(len(X)):
                x_i = X[i]
                d_i = y[i]

                # Вычисление взвешенной суммы 
                net = np.dot(x_i, self.weights) - self.bias 

                # Получение выхода нейрона 
                y_pred = self.activation(net)

                # Проверка соответствия выхода желаемому
                if y_pred != d_i:
                    weights_updated = True # устанавливаем флаг, что на этой эпохе были изменения весов (корекция признаков)

                    # Положительное подкрепление (y=0 → должен быть 1) 
                    if d_i == 1:
                        self.weights += self.learning_rate *  x_i  # Увеличиваем веса активных входов     
                                                                                                                        
                        self.bias -= self.learning_rate       # уменьшаем порог активации

                    # Отрицательное подкрепление (y=1 → должен быть 0)
                    else:
                        self.weights -= self.learning_rate *  x_i  # Уменьшаем веса активных входов 
                        self.bias += self.learning_rate       #  Увеличиваем порог активации
    
            # Проверка завершения обучения
            if not weights_updated:
                print(f"Обучение завершено на эпохе {epoch}")
                break


data = pd.read_csv('diabetes.csv')
data_cleaned = data.dropna()
X = data_cleaned[['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']].values
y = data_cleaned['Outcome'].values

# Нормализация данных 
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели
perceptron = Perceptron(input_size=X.shape[1], learning_rate=0.1, epochs=1000) 
perceptron.train(X_train, y_train)

# Оценка точности
y_pred = perceptron.predict(X_test)
print(f"Точность модели: {accuracy_score(y_test, y_pred):.4f}")