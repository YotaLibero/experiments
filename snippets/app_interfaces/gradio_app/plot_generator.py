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
# 2.  Класс генерации графиков
# --------------------------------------------------------------
class PlotGenerator:
    @staticmethod
    def create_plot(category, obj_idx, start_point, end_point):
        """
        Создает 4 линейных графика для заданной ванны
        """
        # Генерируем данные для графиков
        x = np.linspace(0, 10, 100)
        if category == 2:
            x = np.linspace(1, 11, 100)
        elif category == 5:
            x = np.linspace(5, 15, 100)

        # Создаем фигуру с 4 подграфиками
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Графики для ванны {category}', fontsize=16)

        # График 1: Температура пика
        axes[0, 0].plot(x, start_point * np.sin(x) + start_point/2, 'b-', linewidth=2)
        axes[0, 0].set_title('Температура пика')
        axes[0, 0].set_xlabel('Время')
        axes[0, 0].set_ylabel('Температура')
        axes[0, 0].grid(True)

        # График 2: Солидус
        axes[0, 1].plot(x, end_point * np.cos(x) + end_point/2, 'r-', linewidth=2)
        axes[0, 1].set_title('Солидус')
        axes[0, 1].set_xlabel('Время')
        axes[0, 1].set_ylabel('Температура')
        axes[0, 1].grid(True)

        # График 3: Смешанные данные
        y1 = start_point * np.sin(x) + end_point * np.cos(x)
        axes[1, 0].plot(x, y1, 'g-', linewidth=2)
        axes[1, 0].set_title('Смешанные данные')
        axes[1, 0].set_xlabel('Время')
        axes[1, 0].set_ylabel('Значение')
        axes[1, 0].grid(True)

        # График 4: Сравнение
        axes[1, 1].plot(x, start_point * np.sin(x), 'b--', label='Температура пика')
        axes[1, 1].plot(x, end_point * np.cos(x), 'r--', label='Солидус')
        axes[1, 1].set_title('Сравнение')
        axes[1, 1].set_xlabel('Время')
        axes[1, 1].set_ylabel('Температура')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        return fig

    def get_existing_image(category, obj_idx, start_point, end_point):
        """Возвращает путь к существующей картинке"""
        # Путь к вашей картинке в проекте
        # Предположим, что картинки находятся в папке "images" в проекте
        if category == 2:
            image_path = "dots.png"  # Путь к вашей картинке
        elif category == 5:
            image_path = "dots.png"  # Путь к вашей картинке
        else:
            image_path = "dots.png"  # Путь к картинке по умолчанию

        # Проверяем, существует ли файл
        if os.path.exists(image_path):
            return image_path
        else:
            # Если файл не найден, возвращаем заглушку
            return "dots.png"  # или создайте заглушку
        