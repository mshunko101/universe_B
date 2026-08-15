import numpy as np
from geographiclib.geodesic import Geodesic
from scipy.optimize import minimize

alpha_crit = 0.0084
t = 1.0
rho = 1.0
L_min = 70.0  # отсечка, чтобы не было взрыва вблизи источников

def dist_km(lat1, lon1, lat2, lon2):
    g = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
    return g['s12'] / 1000.0

# Источники: (name, lat, lon, P)
sources = [
    ("Новополоцк", 55.5, 28.6, 2e8),
    ("Гуанчжоу (PRD)", 23.1, 113.3, 5e8),
    ("Рур (Эссен)", 51.4, 7.0, 2e8),
    ("Тяньцзинь (Бохай)", 39.1, 117.2, 4e8),
    ("Иокогама", 35.4, 139.6, 5e8),
    ("Мумбаи", 19.0, 72.8, 2e8),
    ("Детройт", 42.3, -83.0, 2e8),
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

n = len(sources)

def alpha_total_and_components(lat, lon):
    """
    Возвращает:
      - total_alpha
      - список пар: (name_i, name_j, B_ij, alpha_ij)
      - sum_weights (сумма B_ij)
      - weighted_sum (сумма B_ij * alpha_ij)
    """
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
            alpha_ij = B_ij / (rho * (L_eff**5))

            total_alpha += alpha_ij
            sum_weights += B_ij
            weighted_sum += B_ij * alpha_ij

            components.append((name_i, name_j, B_ij, alpha_ij))

    return total_alpha, components, sum_weights, weighted_sum

# 1. Грубый поиск на сетке 5°
lat_grid = np.arange(-90, 91, 5)
lon_grid = np.arange(-180, 181, 5)

best_val = -np.inf
best_lat = None
best_lon = None

for lat in lat_grid:
    for lon in lon_grid:
        # Отсекаем точки, слишком близкие к источникам (чтобы не было «внутри источника»)
        too_close = False
        for _, s_lat, s_lon, _ in sources:
            if dist_km(s_lat, s_lon, lat, lon) < L_min:
                too_close = True
                break
        if too_close:
            continue

        val, _, _, _ = alpha_total_and_components(lat, lon)
        if val > best_val:
            best_val = val
            best_lat = lat
            best_lon = lon

print(f"[Грубый поиск] Точка-кандидат: {best_lat:.1f}°, {best_lon:.1f}°")
print(f"alpha_total (грубо) = {best_val:.6f}")

# 2. Локальная оптимизация вокруг кандидата
def neg_alpha(x):
    lat, lon = x
    val, _, _, _ = alpha_total_and_components(lat, lon)
    return -val

x0 = [best_lat, best_lon]
res = minimize(neg_alpha, x0, method='Nelder-Mead', tol=1e-4)
opt_lat, opt_lon = res.x

# Финальный расчёт в оптимальной точке
total_alpha, components, sum_weights, weighted_sum = alpha_total_and_components(opt_lat, opt_lon)
if sum_weights > 0:
    mean_alpha_weighted = weighted_sum / sum_weights
else:
    mean_alpha_weighted = 0.0

print("\n=== ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ ===")
print(f"Координаты экстремальной точки: широта = {opt_lat:.3f}°, долгота = {opt_lon:.3f}°")
print(f"Суммарный alpha (по парам) = {total_alpha:.6f}")
print(f"Средневзвешенный alpha = {mean_alpha_weighted:.6f}")
print(f"Превышен порог alpha_crit={alpha_crit}? {total_alpha > alpha_crit}")

# Таблица топ-вкладов
components_sorted = sorted(components, key=lambda x: x[3], reverse=True)
print("\nТаблица влияния топ-пар (вклад в alpha):")
print("| Пара источников | B (Дж²/с) | Вклад alpha |")
print("|---|---|---|")
for name_i, name_j, B_ij, alpha_ij in components_sorted[:15]:
    print(f"| {name_i} + {name_j} | {B_ij:.2e} | {alpha_ij:.6f} |")
