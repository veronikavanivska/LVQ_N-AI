# import numpy as np
# from neupy import algorithms
# from sklearn.model_selection import StratifiedKFold
# import hickle as hkl
# from collections.abc import MutableMapping
#
# x, y_t, x_norm, x_n_s, y_t_s = hkl.load('hepatitis.hkl')
# y_t -= 1
# x = x.T
# y_t = np.squeeze(y_t)
# lvqnet = algorithms.LVQ(n_inputs=x.shape[0], n_classes=np.unique(y_t).shape[0], step=0.001)
# # lvqnet = algorithms.LVQ2(n_inputs=x.shape[0], n_classes=np.unique(y_t).shape[0],step=0.001)
# # lvqnet = algorithms.LVQ21(n_inputs=x.shape[0], n_classes=np.unique(y_t).shape[0],step=0.001)
# # http://neupy.com/modules/generated/neupy.algorithms.LVQ3.html
# # lvqnet = algorithms.LVQ3(n_inputs=x.shape[0], n_classes=np.unique(y_t).shape[0],step=0.001)
# lvqnet.train(x, y_t, epochs=1000)
# y = lvqnet.predict(x)
# e = y_t - y
# PK = sum(abs(e) < 0.5) / e.shape[0] * 100
# print("\nPK = %5d" % PK)
#
# data = x
# target = y_t
#
# CVN = 10
# skfold = StratifiedKFold(n_splits=CVN)
# PK_vec = np.zeros(CVN)
#
# for i, (train, test) in enumerate(skfold.split(data, target), start=0):
#     x_train, x_test = data[train], data[test]
#     y_train, y_test = target[train], target[test]
#     # print(i,train,test)
#     lvqnet = algorithms.LVQ(n_inputs=x_train.shape[1], n_classes=np.unique(y_train).shape[0], step=0.001)
#     # lvqnet = algorithms.LVQ2(n_inputs=x.shape[1], n_classes=np.unique(y_t).shape[0],step=0.001)
#     # lvqnet = algorithms.LVQ21(n_inputs=x.shape[1], n_classes=np.unique(y_t).shape[0],step=0.001)
#     # http://neupy.com/modules/generated/neupy.algorithms.LVQ3.html
#     # lvqnet = algorithms.LVQ3(n_inputs=x.shape[1], n_classes=np.unique(y_t).shape[0],step=0.001)
#     lvqnet.train(x_train, y_train, epochs=100)
#     result = lvqnet.predict(x_test)
#
#     n_test_samples = test.size
#     PK_vec[i] = np.sum(result == y_test) / n_test_samples
#
#     print("Test #{:<2}: PK_vec {} test_size {}".format(i, PK_vec[i], n_test_samples))
#
# PK = np.mean(PK_vec)
# print("PK {}".format(PK))
#
#
# import numpy as np
# from neupy import algorithms
# from sklearn.model_selection import StratifiedKFold
# import hickle as hkl
#
# # Load data
# x, y_t, x_norm, x_n_s, y_t_s = hkl.load('hepatitis.hkl')
#
# y_t -= 1
# x = x.T  # Transpose data
# y_t = np.squeeze(y_t)
# # step = 0.01,
# # minstep=0.0001
# # n_updates_to_stepdrop=50000
# # Initialize LVQ network
# lvqnet = algorithms.LVQ(n_inputs=x.shape[0], n_classes=np.unique(y_t).shape[0], step=0.01)
#
# # Train LVQ network
# lvqnet.train(x, y_t, epochs=300)
#
# # Make predictions
# y_pred = lvqnet.predict(x)
#
# # Calculate PK
# e = y_t - y_pred
# PK = (1 - np.sum(np.abs(e) >= 0.5) / e.shape[0]) * 100
# print("\nPK = %5d" % PK)
#
# # Initialize cross-validation
# data = x
# target = y_t
# CVN = 10
# skfold = StratifiedKFold(n_splits=CVN)
#
# # Perform cross-validation
# PK_vec = np.zeros(CVN)
# for i, (train, test) in enumerate(skfold.split(data, target), start=0):
#     x_train, x_test = data[train], data[test]
#     y_train, y_test = target[train], target[test]
#
#     # Initialize LVQ network for each fold
#     lvqnet = algorithms.LVQ(n_inputs=x_train.shape[1], n_classes=np.unique(y_train).shape[0], step=0.01)
#
#     # Train LVQ network for each fold
#     lvqnet.train(x_train, y_train, epochs=300)
#
#     # Make predictions for test set
#     y_pred = lvqnet.predict(x_test)
#
#     # Calculate accuracy for the fold
#     n_test_samples = test.size
#     PK_vec[i] = np.sum(y_pred == y_test) / n_test_samples
#
#     print("Test #{:<2}: PK_vec {} test_size {}".format(i, PK_vec[i], n_test_samples))
#
# # Calculate mean PK over all folds
# PK = np.mean(PK_vec)
# print("PK {}".format(PK))

from sklearn.model_selection import StratifiedKFold
import hickle as hkl
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
from neupy import algorithms

# Odczyt danych z pliku i przypisanie ich do zmiennych
x,y_t, x_norm ,x_n_s, y_t_s = hkl.load('hepatitis.hkl')
y_t -= 1 # Dostosowanie etykiet klas
x = x.T # Transpozycja macierzy x
y_t = np.squeeze(y_t) # Usuwanie jednowymiarowych osi z y_t
data = x_norm.T # Przypisanie transponowanej macierzy 'x_norm' do zmiennej data
target = y_t # Przypisanie wektora zawierającego etykiety do zmiennej target

epoch = 10 # Ustalenie liczby epok do treningu modelu

# Definiowanie metaparametrów
step_vec = np.array([0.5, 0.1, 1e-3, 1e-6, 1e-9, 1e-17, 1e-25]) # Wartości dla parametru 'step'
n_updates_to_stepdrop_vec = np.array([10, 100, 500, 1000, 2500, 5000, 10000]) # Wartości dla parametru 'n_updates_to_stepdrop'
minstep_vec = np.array([0.1, 1e-5, 1e-9, 1e-16, 1e-19, 1e-27]) # Wartości dla parametru 'minstep'

start = timer() # Rozpoczynanie pomiaru czasu
CVN = 10 # Ustalenie liczby foldow dla walidacji krzyżowej
skfold = StratifiedKFold(n_splits=CVN) # Przygotowanie do wykonania walidacji krzyżowej

# Inicjalizacja zmiennych, które przechowują najlepsze parametry
best_step = 0 # Najlepsza wartośc 'step'
best_n_updates_to_stepdrop = 0 # Najlepsza wartość 'n_updates_to_stepdrop'
best_minstep = 0 # Najlepsza wartość 'minstep'
best_PK = 0 # Najlepsza wartość poprawności klasyfikacji
best_PK_minstep = 0 # Najlepsza wartość poprawności klasyfikacji, używana podczas iteracji 'minstep'
temp_minstep = 1e-40 #Tymczasowa wartość 'minstep', użyta dla testowania 'step' oraz 'n_updates_to_stepdrop'

# Macierz przechowująca wartości poprawności klasyfikacji dla kombinacji 'step' oraz 'n_updates_to_stepdrop'
PK_values = np.zeros((len(step_vec),len(n_updates_to_stepdrop_vec)))

# Pętla przechodząca przez różne wartości parametru 'step'
for steps in range(len(step_vec)):
    # Pętla przechodząca przez różne wartości parametru 'n_updates_to_stepdrop'
    for n_updates_to_stepdrop_ in range(len(n_updates_to_stepdrop_vec)):
        print("Step: ", step_vec[steps], "n_updates_to_stepdrop_vec: ",
              n_updates_to_stepdrop_vec[n_updates_to_stepdrop_]) # Wyświetlanie aktualnej kombinacji 'step' i 'n_updates_t _stepdrop'
        # Wektor przechowujący wyniki poprawności klasyfikacji każdego folda
        PK_vec = np.zeros(CVN)
        # Pętla po wszystkich foldach walidacji krzyżowej
        for i, (train, test) in enumerate(skfold.split(data, target),start = 0):
            # Definicja zbioru treningowego i testowego dla danych wejściowych
            x_train , x_test = data[train], data[test]
            # Definicja zbioru treningowego i testowego dla danych wyjściowych
            y_train,  y_test = target[train], target[test]
            # Tworzenie instancji algorytmu LVQ z określonymi parametrami
            lvqnet = algorithms.LVQ3(
                epsilon = 0.3, # Podwójna aktualizacja wag jest wykonana kiedy różnica między dwoma prototypami wynosi 0.3 lub mniej
                n_inputs = x_train.shape[1], # Liczba jednostek wejściowych
                n_classes = np.unique(y_train).shape[0], # Liczba klas w zestawie danych
                step = step_vec[steps], # Współczynnik uczenia
                n_updates_to_stepdrop = n_updates_to_stepdrop_vec[n_updates_to_stepdrop_], # Liczba aktualizacji zmniejszania kroku
                minstep = temp_minstep) # Testowy 'minstep'
            lvqnet.train(x_train,y_train, epochs=epoch) # Trening sieci na zbiorze treningowym
            result = lvqnet.predict(x_test) # Przewidywanie etykiet dla zestawu testowego
            n_test_samples = test.size # Obliczenie liczby próbek w zestawie treningowym
            PK_vec[i] = (np.sum(result == y_test) / n_test_samples) * 100 # Obliczanie precyzji poprawności klasyfikacji

            # Wypisanie na ekran aktualnych danych dla biezącego podziału kroswalidacji
            print("Test #{:<2}: PK_vec {} test_size {}".format(i, PK_vec[i],n_test_samples))

        PK = np.mean(PK_vec) # Średnia poprawność klasyfikacji dla wszystkich podziałów kroswalidacji
        PK_values[steps,n_updates_to_stepdrop_] = PK # Przypisanie średniej PK do tablicy PK_values na odpowiadające kombinacje
        # 'step' oraz 'n_updates_to_stepdrop'

        # Czy obliczona średnia jest większa niż obecnie najlepsze PK
        if PK > best_PK:
            best_PK = PK # Jeżeli tak, przypisywana jest najwyższa średnia wartość PK do 'best_PK'
            best_step = step_vec[steps] # Przypisanie najlepszego kroku, dla najwyższego PK
            best_n_updates_to_stepdrop = n_updates_to_stepdrop_vec[n_updates_to_stepdrop_] # Najlepsze 'n_updates_to_stepdrop'

        print("Srednie PK dla wykonanej iteracji: {}\n".format(PK))
print("Najlepsze parametry:\nPoprawnosc Klasyfikacji: {}\nstep: {}\nn_updates_to_stepdrop: {}".format(best_PK,best_step,best_n_updates_to_stepdrop))
print("Czas wykonania:", timer()-start) # Wyświetlanie najlepszych parametrów

# Rysowanie 3D wykresu
fig = plt.figure(figsize = (8,8)) # Tworzy nową figurę do rysowania wykresu, o rozmiarze 8x8
ax = fig.add_subplot(111, projection='3d') # Tworzy osie do rysowania 3D na objekcie figury. '111' , oznacza, że jest utworzony pojedynczy wykres na figurze
X,Y = np.meshgrid(np.log10(step_vec), n_updates_to_stepdrop_vec) #Tworzy siatkę współrzędnych dla wykresów 3D dla 'step_vec', którego wartości są logarytmowane i 'n_updates_to_stepdrop'
surf = ax.plot_surface(X,Y,PK_values.T, cmap='viridis') # Tworzy powierzchnie 3D używając X,Y i transpozycji macierzy PK_values jako z. 'Viridis' to mapa kolorów kolorujących przestrzeń
ax.set_xlabel('log10(step)') # Ustawienie etykiety osi x jako 'log10(step)'
ax.set_ylabel('n_updates_to_stepdrop') # Ustawienie etykiety osi y na 'n_updates_to_stedrop'
ax.set_zlabel('PK') # Ustawienie etykiety osi z na 'PK'
plt.show() # Wyświetlanie utworzonego wykresu

start = timer() #Rozpoczęcie pomiaru czasu

PK_values_minstep = np.zeros(len(minstep_vec)) # Tablica używana do przechowywania poprawności klasyfikacji zależnej od 'minstep'
# Pętla przechodząca przez różne wartości parametru 'minstep'
for minstep_index in range(len(minstep_vec)):
    print("Minstep: ",minstep_vec[minstep_index]) # Wyświetlenie aktualnej wartości 'minstep'
    PK_vec = np.zeros(CVN) # Wektor przechowujący wyniki poprawności klasyfikacji kazdego folda
    # Pętla po wszystkich foldach walidacji krzyżowej
    for i,(train,test) in enumerate(skfold.split(data, target), start = 0):
        x_train,x_test = data[train], data[test] # Definicja zbioru treningowego i testowego dla danych wejściowych
        y_train,y_test = target[train], target[test] # Definicja zbioru treningowego i testowego dla danych wyjściowych
        # Tworzenie instancji algorytmu LVQ z określonymi parametrami
        lvqnet = algorithms.LVQ3(
            epsilon = 0.3,
            n_inputs = x_train.shape[1],# Liczba jednostek wejściowych
            n_classes = np.unique(y_train).shape[0],# Liczba klas w zestawie danych
            step = best_step, # Learning rate dla którego osiągnięto najwyższe PK
            n_updates_to_stepdrop = best_n_updates_to_stepdrop, # Liczba aktualizacji zmniejszania kroku dla którego osiągnięto najwyższe PK
            minstep = minstep_vec[minstep_index] # Minimalna wartość kroku
        )
        lvqnet.train(x_train,y_train,epochs=epoch) # Trening sieci na zbiorze treningowym
        result = lvqnet.predict(x_test) # Przewidywanie etykiet dla zestawu testowego
        n_test_samples = test.size  # Obliczenie liczby próbek w zestawie treningowym
        PK_vec[i] = (np.sum(result == y_test)/n_test_samples) * 100 # Obliczanie precyzji poprawności klasyfikacji

        print("Test #{:<2}: PK_vec {} test_size {} minstep: {}".format(i,PK_vec[i],n_test_samples,minstep_vec[minstep_index]))

    PK = np.mean(PK_vec) # Średnia poprawność klasyfikacji dla wszystkich foldów
    PK_values_minstep[minstep_index] = PK # Średnia poprawność klasyfikacji dla aktualnej wartości 'minstep'
    # Czy obliczona średnia jest większa niż obecnie najlepsze PK
    if PK > best_PK_minstep:
        best_PK_minstep = PK # Jeżeli tak to aktualizowana jest najlepsza wartość PK
        best_minstep = minstep_vec[minstep_index] # Aktualizacja 'minstep' dla najlepszej poprawności

# Rysowanie wykresu zależności między 'minstep' i PK
plt.figure(figsize=(8,6)) # Tworzy nową figurę do rysowania wykresu o rozmiarze 8x6
plt.plot(minstep_vec,PK_values_minstep, 'o-') # Rysuje liniowy wykres 'minstep_vec' na osi x i 'PK_values_minstep' na osi y. Punkty danych są okręgami, a linia jest ciągła
plt.xscale('log') # Skala osi x jest logarytmiczna
plt.title('Zależność PK od minstep') # Ustalanie tytułu
plt.xlabel('minstep') # Ustalanie etukiety osi x na 'minstep'
plt.ylabel('PK') # Ustalenie etykiety osi y na 'PK'
plt.grid(True) # Dodawanie siatki do wykresu
plt.show() #Wyświetlanie utworzonego wykresu


print("Najlepsze parametry:\nPoprawność Klasyfikacji: {} minstep: {}".format(best_PK_minstep,best_minstep))
print("Czas wykonania:", timer()-start) # Wyświetlenie najlepszych parametrów dla eksperymetu
