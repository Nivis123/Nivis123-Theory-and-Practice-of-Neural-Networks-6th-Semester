import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy as libentropy

print("=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА: ПРЕПРОЦЕССИНГ ДАТАСЕТА")
print("=" * 60)


print("\n" + "=" * 50)
print("ШАГ 1: Загрузка и первичный анализ данных")
print("=" * 50)

df = pd.read_csv('Laptop_price.csv')

print("\nПервые 5 строк:")
print(df.head())

print("\nИнформация о данных:")
print(df.info())

print("\nОписательная статистика:")
print(df.describe())

print("\nПропуски в каждом столбце:")
print(df.isnull().sum())

print("\nКоличество дубликатов:", df.duplicated().sum())



print("\n" + "=" * 50)
print("ШАГ 2: Корреляционный анализ")
print("=" * 50)

numeric_cols = ['Processor_Speed', 'RAM_Size', 'Storage_Capacity',
                'Screen_Size', 'Weight', 'Price']

corr_matrix = df[numeric_cols].corr()


plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, fmt='.3f')
plt.title('Корреляционная матрица числовых признаков', fontsize=14)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nАнализ корреляций (порог |r| > 0.6):")
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.6:
            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

if high_corr:
    print("Найдены пары с высокой корреляцией:")
    for pair in high_corr:
        print(f"  {pair[0]} и {pair[1]}: {pair[2]:.3f}")
else:
    print("Не обнаружено пар признаков с корреляцией > 0.7")

print("\nВывод: Так как сильных корреляций нет, все признаки можно оставить для дальнейшего анализа.")


print("\n" + "=" * 50)
print("ШАГ 3: Подготовка данных для расчёта Gain Ratio")
print("=" * 50)

n_bins = 5

print(f"\nДискретизация непрерывных признаков на {n_bins} интервалов:")

df['Processor_Speed_binned'] = pd.cut(df['Processor_Speed'], bins=n_bins, labels=False)
print(f"  Processor_Speed: мин={df['Processor_Speed'].min():.2f}, макс={df['Processor_Speed'].max():.2f}")
print(f"    После дискретизации: уникальные значения = {sorted(df['Processor_Speed_binned'].unique())}")

df['Screen_Size_binned'] = pd.cut(df['Screen_Size'], bins=n_bins, labels=False)
print(f"  Screen_Size: мин={df['Screen_Size'].min():.2f}, макс={df['Screen_Size'].max():.2f}")
print(f"    После дискретизации: уникальные значения = {sorted(df['Screen_Size_binned'].unique())}")

df['Weight_binned'] = pd.cut(df['Weight'], bins=n_bins, labels=False)
print(f"  Weight: мин={df['Weight'].min():.2f}, макс={df['Weight'].max():.2f}")
print(f"    После дискретизации: уникальные значения = {sorted(df['Weight_binned'].unique())}")

print("\nДискретизация целевой переменной Price на 3 класса:")
df['Price_class'] = pd.qcut(df['Price'], q=3, labels=['low', 'medium', 'high'])
print("\nРаспределение классов цены:")
print(df['Price_class'].value_counts())
print("\nСтатистика по классам:")
print(df.groupby('Price_class')['Price'].describe())



print("\nРучное кодирование категориальных переменных:")

def manual_encode(series):
    unique_vals = series.unique()
    val_to_int = {val: i for i, val in enumerate(sorted(unique_vals))}
    encoded = series.map(val_to_int).values
    print(f"  {series.name}: {dict(sorted(val_to_int.items()))}")
    return encoded


brand_encoded = manual_encode(df['Brand'])
price_class_encoded = manual_encode(df['Price_class'])


print(f"\nЦелевая переменная закодирована: 0=low, 1=medium, 2=high")

df['Brand_encoded'] = brand_encoded
df['Price_class_encoded'] = price_class_encoded


print("\n" + "=" * 50)
print("ШАГ 4: Расчёт Gain Ratio (собственная реализация)")
print("=" * 50)


def entropy(y):
    if len(y) == 0:
        return 0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def split_info(x):
    return entropy(x)


def info_gain(x, y):
    H_parent = entropy(y)

    unique_vals = np.unique(x)
    H_child = 0
    for val in unique_vals:
        mask = (x == val)
        if np.sum(mask) == 0:
            continue
        y_sub = y[mask]
        w = len(y_sub) / len(y)
        H_child += w * entropy(y_sub)

    return H_parent - H_child


def gain_ratio(x, y):
    ig = info_gain(x, y)
    si = split_info(x)
    if si == 0:
        return 0
    return ig / si


features_for_gain = {
    'Brand': df['Brand_encoded'].values,
    'Processor_Speed': df['Processor_Speed_binned'].values,
    'RAM_Size': df['RAM_Size'].values,
    'Storage_Capacity': df['Storage_Capacity'].values,
    'Screen_Size': df['Screen_Size_binned'].values,
    'Weight': df['Weight_binned'].values
}

y_target = df['Price_class_encoded'].values

print("\nРасчёт Gain Ratio для всех признаков:")
gain_ratios = {}

for name, x in features_for_gain.items():
    gr = gain_ratio(x, y_target)
    gain_ratios[name] = gr
    print(f"  {name}: {gr:.6f}")

sorted_gr = sorted(gain_ratios.items(), key=lambda item: item[1], reverse=True)

print("\n" + "=" * 50)
print("РЕЗУЛЬТАТЫ: Признаки, отсортированные по Gain Ratio (ручной расчёт)")
print("=" * 50)
for i, (name, gr) in enumerate(sorted_gr, 1):
    print(f"{i}. {name}: {gr:.6f}")


print("\n" + "=" * 50)
print("ШАГ 5: Проверка энтропии библиотечной функцией")
print("=" * 50)


def entropy_lib(labels):
    if len(labels) == 0:
        return 0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return libentropy(probs) / np.log(2)

print("\nЭнтропия признаков (SplitInfo):")
for name, x in features_for_gain.items():
    si_manual = split_info(x)
    si_lib = entropy_lib(x)
    print(f"  {name}: ручная={si_manual:.6f}, библ={si_lib:.6f}, разница={abs(si_manual - si_lib):.2e}")

print("\n" + "=" * 50)
print("ШАГ 6: Визуализация и отбор признаков")
print("=" * 50)

results_df = pd.DataFrame({
    'Feature': [name for name, _ in sorted_gr],
    'Gain_Ratio': [gr for _, gr in sorted_gr]
})

plt.figure(figsize=(12, 7))
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#6A4E3A']
bars = plt.bar(range(len(results_df)), results_df['Gain_Ratio'], color=colors[:len(results_df)])
plt.xticks(range(len(results_df)), results_df['Feature'], rotation=45, ha='right', fontsize=11)
plt.xlabel('Признаки', fontsize=12)
plt.ylabel('Gain Ratio', fontsize=12)
plt.title('Важность признаков по метрике Gain Ratio', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, results_df['Gain_Ratio'])):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
             f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

mean_gr = np.mean(list(gain_ratios.values()))
threshold = mean_gr

print(f"\nПорог отбора (среднее значение): {threshold:.6f}")
selected_features = [name for name, gr in gain_ratios.items() if gr > threshold]

print(f"\nОтобрано признаков: {len(selected_features)} из {len(gain_ratios)}")
print("Выбранные признаки:")
for i, feat in enumerate(selected_features, 1):
    gr_value = gain_ratios[feat]
    print(f"  {i}. {feat}: {gr_value:.6f}")


print("ШАГ 8: ВЫВОДЫ ПО ЛАБОРАТОРНОЙ РАБОТЕ")
print("=" * 50)


print("\nОТОБРАННЫЕ ПРИЗНАКИ:")
for i, feat in enumerate(selected_features, 1):
    print(f"{i}. {feat}: Gain Ratio = {gain_ratios[feat]:.6f}")

columns_to_save = ['Brand', 'Processor_Speed', 'RAM_Size', 'Storage_Capacity',
                   'Screen_Size', 'Weight', 'Price', 'Price_class']
for feat in selected_features:
    if feat in df.columns:
        columns_to_save.append(feat)
    elif feat + '_binned' in df.columns:
        columns_to_save.append(feat + '_binned')

columns_to_save = list(dict.fromkeys(columns_to_save))

output_df = df[columns_to_save].copy()
output_df.to_csv('Laptop_price_processed.csv', index=False)
print(f"\nОбработанные данные сохранены в 'Laptop_price_processed.csv'")
print(f"Сохранены колонки: {', '.join(columns_to_save)}")