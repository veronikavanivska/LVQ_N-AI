import hickle as hkl
import numpy as np
import matplotlib.pyplot as plt # Import bibliotek
filename = open("hepatitis.txt") # Otwieranie pliku z danymi
data = np.loadtxt(filename, delimiter=',', dtype=str) # Wczytanie danych z pliku jako tablica numpy z elementami typu string

data = data[(data=='?').any(axis=1)==0] #usunięcie rekordów z brakującymi danymi

x = data[:, 1:].astype(float).T # Ustawienie cech wejściowych
y_t = data[:,0].astype(float) #ustawienie wyjścia pożądanego
y_t = y_t.reshape(1,y_t.shape[0]) #zmiana na jednowierszową tablicę

# Wypisanie zakresu wartości dla każdej cechy przed normalizacją
print(np.transpose([np.array(range(x.shape[0])), x.min(axis=1),
x.max(axis=1)]))

x_min = x.min(axis=1) # Minimalne wartości dla każdej cechy
x_max = x.max(axis=1) # Maksymalne wartości dla każdej cechy
x_norm_max = 1 # Maksymalna wartość po normalizacji
x_norm_min = -1 # Minimalna wartość po normalizacji
x_norm = np.zeros(x.shape) # Inicjalizacja znormalizowanej tablicy z zerami

# Normalizacja danych do zakresu [-1, 1]
for i in range(x.shape[0]):
    x_norm[i,:] = (x_norm_max-x_norm_min)/(x_max[i]-x_min[i])* \
    (x[i,:]-x_min[i]) + x_norm_min

print(np.transpose([np.array(range(x.shape[0])), x_norm.min(axis=1),
x_norm.max(axis=1)])) #Wypisanie znormalizowanych zbiorów

y_t_s_ind = np.argsort(y_t)  # Indeksy sortujące y_t
x_n_s = np.zeros(x.shape)  # Inicjalizacja posortowanej tablicy cech
y_t_s = np.zeros(y_t.shape) # Inicjalizacja posortowanej tablicy wyjść

# Sortowanie danych na podstawie posortowanych wyjść
for i in range(x.shape[1]):
    y_t_s[0,i] = y_t[0,y_t_s_ind[0,i]] # Przypisanie posortowanych wartości wyjść
    x_n_s[:,i] = x_norm[:,y_t_s_ind[0,i]] # Przypisanie posortowanych cech

plt.plot(y_t_s[0]) # Wykres posortowanego zbioru wyjść
plt.show()  # Wyświetlenie wykresu

# Zapisanie wyników do pliku hepatitis2.hkl
hkl.dump([x,y_t,x_norm,x_n_s,y_t_s],"hepatitis2.hkl")
# Wczytanie wyników z pliku hepatitis2.hkl
x,y_t,x_norm,x_n_s,y_t_s = hkl.load("hepatitis2.hkl")
