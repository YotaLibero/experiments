import os
import gradio as gr
import pandas as pd
import numpy as np
import time
import threading
import random
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Устанавливаем backend перед созданием графиков
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# 3.  Класс управления динамическими списками
# --------------------------------------------------------------
class AutoDropdownUpdater:
    """Генерирует произвольный список чисел каждые N секунд."""
    def __init__(self, data_provider):
        self.data_provider = data_provider
        self.dt_choices = ["2024-03-16 04:32:55", "2024-04-02 15:04:23"]  # [datetime(2024, 3, 16, 4, 32, 55), datetime(2024, 4, 2, 15, 4, 23)]   #        
        self.choices = [1, 2]        
        self.is_running = True
        self.lock = threading.Lock()

    def backgroundworker(self):
        """Работает в отдельном потоке и меняет self.choices."""
        while self.is_running:
            # случайный набор от 1 до 9 (можно менять диапазон)
            new_choices = self.data_provider.get_event_choices() # list(range(1, random.randint(3, 10)))
            new_dt_choices = self.data_provider.get_datetime_choices()            
            with self.lock:
                self.choices = new_choices
                self.dt_choices = new_dt_choices
            print(f"[AutoDropdown] Новые варианты → {new_choices}")
            time.sleep(10000)               # обновляем раз в 10 сек

    def get_from_db(self): # ф-ция для чтения ивентов из бд по ванне за день
        pass

    def start(self):
        threading.Thread(target=self.backgroundworker, daemon=True).start()

    def get_choices(self) -> list[int]:
        """Возврат текущего списка, защищённый блокировкой."""
        with self.lock:
            return self.choices.copy()

    def get_dt_choices(self) -> list[str]:
        """Возврат текущего списка, защищённый блокировкой."""
        with self.lock:
            return self.dt_choices.copy()
        