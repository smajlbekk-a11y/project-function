Activation Analysis: Жылдам Бастау

Бұл жоба MLP моделінің өнімділігін 3 активация функциясы (sigmoid, tanh, relu) және 4 оқыту жылдамдығы (0.1 ден 0.0001 дейін) бойынша салыстырады.

1. Қолдану

Қадам 1: Барлық Эксперименттерді Оқыту және Нәтижелерді Біріктіру

Барлық 12 комбинацияны (3x4) автоматты түрде іске қосу және нәтижелерді бір файлға біріктіру.

# Барлық оқыту процестерін іске қосу және нәтижелерді results/combined_lr_history.csv файлына жинау
!python src/run_all_experiments.py


Қадам 2: Нәтижелерді Визуализациялау

Біріктірілген файл негізінде негізгі графиктерді құру.

1. Валидация Дәлдігі vs Learning Rate (LR)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('results/combined_lr_history.csv')

# Дәлдікті LR және Активация бойынша көрсету
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='epoch', y='val_acc', hue='learning_rate', style='activation')
plt.title("Validation Accuracy vs Learning Rate")
plt.show()


2. Тиімділік (Accuracy per second) Бағасы

# Тиімділікті есептеу
data['efficiency'] = data['val_acc'] / data['epoch_time_sec']

# Тиімділікті Learning Rate және Activation бойынша салыстыру
plt.figure(figsize=(10, 6))
sns.barplot(data=data, x='learning_rate', y='efficiency', hue='activation')
plt.title("Efficiency (Accuracy per second) by LR and Activation")
plt.show()


2. Қорытынды

Нәтижелер: Барлық нәтижелер (CSV, логтар, графиктер) results/ папкасында сақталады.
