import numpy as np
from geographiclib.geodesic import Geodesic
from scipy.optimize import minimize

alpha_crit = 0.0084
t = 1.0
rho = 1.0
L_min = 70.0

# Параметры "управляющего центра" (Детройт)
eta = 50.0       # Сила модуляции (подбирается, чтобы объяснить всплеск)
lambda_ctrl = 2000.0  # Радиус действия управляющего поля в км

def dist_km(lat1, lon1, lat2, lon2):
    g = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
    return g['s12'] / 1000.0

sources = [
    ("Новополоцк", 55.5, 28.6, 2e8),
    ("Гуанчжоу (PRD)", 23.1, 113.3, 5e8),
    ("Рур (Эссен)", 51.4, 7.0, 2e8),
    ("Тяньцзинь (Бохай)", 39.1, 117.2, 4e8),
    ("Иокогама", 35.4, 139.6, 5e8),
    ("Мумбаи", 19.0, 72.8, 2e8),
    ("Детройт", 42.3, -83.0, 2e8),  # <-- Это наш "центр управления"
    ("Шанхай", 31.2, 121.5, 5e8),
    ("Сеул", 37.5, 126.9, 4e8),
    ("Стамбул", 41.0, 28.9, 2e8),
    ("Мехико", 19.4, -99.1, 2e8),
    ("Лондон", 51.5, -0.1, 2e8),
    ("Сан-Паулу", -23.5, -46.6, 2e8),
    ("Каир", 30.0, 31.2, 2e8),
    ("Джакарта", -6.2, 106.8, 2e8),
    ("Хьюстон", 29.7, -95.4, 2e8),
    ("Дубай", 25.2, 55.3, 2e8),
    ("Калькутта", 22.5, 88.3, 2e8),
    ("Торонто", 43.7, -79.4, 2e8),
    ("Санкт-Петербург", 59.9, 30.3, 2e8)
]

# Найдём индекс Детройта
detroit_idx = None
for i, (name, _, _, _) in enumerate(sources):
    if name == "Детройт":
        detroit_idx = i
        break

n = len(sources)

def alpha_total_with_control(lat, lon):
    total_alpha = 0.0
    sum_weights = 0.0
    weighted_sum = 0.0
    components = []

    for i in range(n):
        for j in range(i+1, n):
            name_i, lat_i, lon_i, P_i = sources[i]
            name_j, lat_j, lon_j, P_j = sources[j]

            L_iX = dist_km(lat_i, lon_i, lat, lon)
            L_jX = dist_km(lat_j, lon_j, lat, lon)
            L_avg = (L_iX + L_jX) / 2.0
            L_eff = max(L_avg, L_min)

            B_ij = P_i * P_j * t

            # --- НОВАЯ ЧАСТЬ: Коэффициент модуляции от Детройта ---
            gamma = 1.0
            if detroit_idx is not None:
                # Расстояние от источников пары до Детройта
                d_i_D = dist_km(lat_i, lon_i, sources[detroit_idx][1], sources[detroit_idx][2])
                d_j_D = dist_km(lat_j, lon_j, sources[detroit_idx][1], sources[detroit_idx][2])
                
                # Среднее расстояние пары до центра управления
                d_pair_D = (d_i_D + d_j_D) / 2.0
                
                # Экспоненциальное затухание влияния центра
                gamma = 1.0 + eta * np.exp(-d_pair_D / lambda_ctrl)
            # ------------------------------------------------------

            alpha_ij = gamma * B_ij / (rho * (L_eff**5))

            total_alpha += alpha_ij
            sum_weights += B_ij
            weighted_sum += B_ij * alpha_ij

            components.append((name_i, name_j, B_ij, alpha_ij, gamma))

    return total_alpha, components, sum_weights, weighted_sum

# Поиск экстремума (упрощённо, можно добавить сетку)
def neg_alpha(x):
    lat, lon = x
    val, _, _, _ = alpha_total_with_control(lat, lon)
    return -val

# Стартуем из Детройта, чтобы найти максимум
x0 = [42.3, -83.0]
res = minimize(neg_alpha, x0, method='Nelder-Mead', tol=1e-4)
opt_lat, opt_lon = res.x

total_alpha, components, sum_weights, weighted_sum = alpha_total_with_control(opt_lat, opt_lon)
mean_alpha_weighted = weighted_sum / sum_weights if sum_weights > 0 else 0.0

print(f"Координаты: {opt_lat:.3f}°, {opt_lon:.3f}°")
print(f"Суммарный alpha (с управлением): {total_alpha:.6f}")
print(f"Средневзвешенный alpha: {mean_alpha_weighted:.6f}")

# Вывод топ-пар с коэффициентом gamma
components_sorted = sorted(components, key=lambda x: x[3], reverse=True)
print("\nТоп-пары (с учётом влияния Детройта):")
print("| Пара | B | alpha вклад | gamma |")
print("|---|---|---|---|")
for name_i, name_j, B, alpha, gamma in components_sorted[:10]:
    print(f"| {name_i}+{name_j} | {B:.2e} | {alpha:.6f} | {gamma:.2f} |")
