import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=1000, stop_threshold=1e-3):
        # Инициализация параметров модели
        self.weights = np.random.randn(input_size) * 0.01  # Инициализация весов малыми случайными значениями
        self.bias = np.random.randn() * 0.01               # Инициализация смещения
        self.learning_rate = learning_rate                
        self.epochs = epochs                             
        self.stop_threshold = stop_threshold               # Порог для ранней остановки

    def activation(self, x):
        return 1 / (1 + np.exp(-x))
    
    def predict(self, X):
        
        z = np.dot(X, self.weights) + self.bias        # Вычисление взвешенной суммы 
        return (self.activation(z) > 0.5).astype(int)  # Пороговая функция
    
    def train(self, X, y):
       
        q = len(y)  # Размер обучающей выборки
        for epoch in range(self.epochs):
            total_error = 0.0                            # Общая ошибка (E_общ)
            delta_weights = np.zeros_like(self.weights)  # Накопление изменений весов 
            delta_bias = 0.0                             # Накопление изменений смещения
            
            
            for i in range(q):
                # Получение i-го обучающего примера
                x_i = X[i]  # Входной вектор 
                d_i = y[i]  # Желаемый выход 
                
                # Прямое распространение
                net = np.dot(x_i, self.weights) + self.bias # Вычисление взвешенной суммы
                y_pred = self.activation(net)               # Активация 
                
                # Вычисление ошибки
                error = d_i - y_pred               # Ошибка для j-го нейрона (d_j - y_j)
                total_error += 0.5 * (error ** 2)  # Накопление ошибки 
                
                # Вычисление градиента 
                delta = error * y_pred * (1 - y_pred) 
                
                # Накопление изменений весов 
                delta_weights += delta * x_i  
                delta_bias += delta          
            
            # Усреднение градиентов по всем обучающим примерам
            delta_weights /= q  
            delta_bias /= q     
            
            # Обновление весов 
            self.weights += self.learning_rate * delta_weights  
            self.bias += self.learning_rate * delta_bias        
            
            # Вычисление средней ошибки 
            avg_error = total_error / q  
            
            # Вывод ошибки каждые 100 эпох
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Error: {avg_error:.6f}")
            
            # Проверка критерия останова
            if avg_error < self.stop_threshold:
                print(f"Ранняя остановка на эпохе {epoch}, Error: {avg_error:.6f}")
                break

# Подготовка данных
data = pd.read_csv('diabetes.csv')  # Загрузка данных

data_cleaned = data.dropna()

# Выбор признаков
X = data_cleaned[['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']].values  
y = data_cleaned['Outcome'].values 
 
# Нормализация данных 
scaler = StandardScaler()
X = scaler.fit_transform(X)  

# Разделение данных 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация модели
perceptron = Perceptron(
    input_size=X.shape[1],        # Размерность входного вектора 
    learning_rate=0.1,           
    epochs=1000,                 
    stop_threshold=0.01           # Порог ошибки для остановки
)

# Обучение модели 
perceptron.train(X_train, y_train)

# Оценка модели
y_pred = perceptron.predict(X_test)  # Предсказание на тестовых данных
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")  # Вычисление точности