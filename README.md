# Neural Network Training & Analysis

Жоба әртүрлі активация функциялары (`sigmoid`, `tanh`, `relu`) мен learning rate параметрлерін тексеріп, нейрондық желінің оқу процесін талдайды. Нәтижелер CSV файлдарға сақталып, графиктер арқылы салыстырылады.

## Қолдану

### Нейрондық желіні оқу
```bash
python src/train.py --activation sigmoid
python src/train.py --activation tanh
python src/train.py --activation relu
Нәтижелерді біріктіру және салыстыру
bash
Копировать код
python src/compare_plots.py
python src/combine_results.py
Learning Rate және Activation әсерін графикте көрсету
python
Копировать код
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('results/combined_lr_history.csv')
sns.lineplot(data=data, x='epoch', y='val_acc', hue='learning_rate', style='activation')
plt.show()
Нәтижелер
Validation Accuracy және Train Loss динамикасы

Орташа көрсеткіштерді салыстыру

Efficiency (Accuracy per second)

Барлық нәтижелер results/ папкасында сақталады.
