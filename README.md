## Diploma Thesis Repository
**Repozytorium kodu pracy dyplomowej (magisterskiej)**

Zawiera pliki skryptów języka Python, oraz strukturę folderów danych projektu aplikacji wykonanej w ramach pracy dyplomowej pt. "Klasyfikacja roślin z wykorzystaniem sztucznej inteligencji"

**preparation.py**
Pozwala na obróbkę zbiorów danych (datasets/) pod kątem kompatybliności formatów z narzędziami biblioteki keras. Dostarcza informacji statystycznych o zbiorach danych.

**training.py**
Implementuje proces uczenia maszynowego na modelu 'zaawansowanym'. Pozwala na wybór zbioru danych do załadowania, buduje model i przeprowadza proces uczenia. Zapisuje wagi warstw modelu i dane statystyczne (saves/).

**training_restore.py**
Pomocniczy plik pozwalający na kontynuowanie zakończonej lub przerwanej sesji uczenia. Tworzy nową zapisy modelu i nowe dane statystyczne.

**inference.py**
Korzysta z zapisanych modeli sieci neuronowych i przeprowadza proces inferencji na nowych danych (test_images/).

**plotting.py**
Zawiera funkcje rysujące wykresy statystyk procesu nauczania na podstawie historii, oraz przedstawianie modelu.