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

# from plot_generator import PlotGenerator
# from auto_dropdown_updater import AutoDropdownUpdater
# from ui_controller import UIController
# from dta_app import DTAApp
# --------------------------------------------------------------
# 1.  Класс заглушек данных (может быть заменен на реальное API)
# --------------------------------------------------------------
class DataProvider:
    """Класс заглушек данных, может быть расширен для подключения к API"""

    def __init__(self):

        self.database_category = {
            2: [
                {'id': 1, 'event_id': 231, 'datetime': datetime(2025, 3, 1, 2, 34, 21), 'record': 566, 'start_point': 642, 'end_point': 1800,
                 'object_index': pd.DataFrame(
                                                np.array([
                                                    ['name_1', 'name_2', 'name_3', 'name_4', 'name_5', 'name_6', 'name_7'],
                                                    [1, 2, 3, 4, 5, 6, 7],
                                                    [1, 2, 31, 2, 2, 4, 28],
                                                    [1, 2, 31, 2, 2, 4, 28],
                                                ]).T,
                                                columns=["index", "OBJ1", "OBJ2", "OBJ3"],
                                            ),},
                {'id': 2, 'event_id': 232, 'datetime': datetime(2025, 3, 1, 13, 21, 3), 'record': 567, 'start_point': 550, 'end_point': 2000, 
                 'object_index': pd.DataFrame(
                                                np.array([
                                                    ['name_1', 'name_2', 'name_3', 'name_4', 'name_5', 'name_6', 'name_7'],
                                                    [1, 2, 3, 4, 5, 6, 7],
                                                    [1, 2, 31, 2, 2, 4, 28],
                                                    [1, 2, 31, 2, 2, 4, 28],
                                                ]).T,
                                                columns=["index", "OBJ1", "OBJ2", "OBJ3"],
                                            ),},
                {'id': 3, 'event_id': 233, 'datetime': datetime(2025, 3, 1, 22, 17, 8), 'record': 568, 'start_point': 700, 'end_point': 1950, 
                 'object_index': pd.DataFrame(
                                                np.array([
                                                    ['name_1', 'name_2', 'name_3', 'name_4', 'name_5', 'name_6', 'name_7'],
                                                    [1, 2, 3, 4, 5, 6, 7],
                                                    [1, 2, 31, 2, 2, 4, 28],
                                                    [1, 2, 31, 2, 2, 4, 28],
                                                ]).T,
                                                columns=["index", "OBJ1", "OBJ2", "OBJ3"],
                                            ),},
                {'id': 4, 'event_id': 234, 'datetime': datetime(2025, 3, 3, 2, 56, 9), 'record': 569, 'start_point': 640, 'end_point': 1900, 
                 'object_index': pd.DataFrame(
                                                np.array([
                                                    ['name_1', 'name_2', 'name_3', 'name_4', 'name_5', 'name_6', 'name_7'],
                                                    [1, 2, 3, 4, 5, 6, 7],
                                                    [1, 2, 31, 2, 2, 4, 28],
                                                    [1, 2, 31, 2, 2, 4, 28],
                                                ]).T,
                                                columns=["index", "OBJ1", "OBJ2", "OBJ3"],
                                            ),},
                {'id': 5, 'event_id': 235, 'datetime': datetime(2025, 3, 2, 15, 6, 37), 'record': 570, 'start_point': 590, 'end_point': 1700, 
                 'object_index': pd.DataFrame(
                                                np.array([
                                                    ['name_1', 'name_2', 'name_3', 'name_4', 'name_5', 'name_6', 'name_7'],
                                                    [1, 2, 3, 4, 5, 6, 7],
                                                    [1, 2, 31, 2, 2, 4, 28],
                                                    [1, 2, 31, 2, 2, 4, 28],
                                                ]).T,
                                                columns=["index", "OBJ1", "OBJ2", "OBJ3"],
                                            ),}
            ],
            5: [
                {'id': 1, 'event_id': 572, 'datetime': datetime(2025, 5, 1, 2, 34, 21), 'record': 566, 'start_point': 645, 'end_point': 1870,
                 'object_index': pd.DataFrame(
                                                np.array([
                                                    ['name_1', 'name_2', 'name_3', 'name_4', 'name_5', 'name_6', 'name_7'],
                                                    [1, 2, 3, 4, 5, 6, 7],
                                                    [1, 2, 31, 2, 2, 4, 28],
                                                    [1, 2, 31, 2, 2, 4, 28],
                                                ]).T,
                                                columns=["index", "OBJ1", "OBJ2", "OBJ3"],
                                            ),},
                {'id': 2, 'event_id': 573, 'datetime': datetime(2025, 5, 1, 13, 21, 3), 'record': 567, 'start_point': 650, 'end_point': 2200, 
                 'object_index': pd.DataFrame(
                                                np.array([
                                                    ['name_1', 'name_2', 'name_3', 'name_4', 'name_5', 'name_6', 'name_7'],
                                                    [1, 2, 3, 4, 5, 6, 7],
                                                    [1, 2, 31, 2, 2, 4, 28],
                                                    [1, 2, 31, 2, 2, 4, 28],
                                                ]).T,
                                                columns=["index", "OBJ1", "OBJ2", "OBJ3"],
                                            ),},
                {'id': 3, 'event_id': 574, 'datetime': datetime(2025, 5, 1, 22, 17, 8), 'record': 568, 'start_point': 750, 'end_point': 2050, 
                 'object_index': pd.DataFrame(
                                                np.array([
                                                    ['name_1', 'name_2', 'name_3', 'name_4', 'name_5', 'name_6', 'name_7'],
                                                    [1, 2, 3, 4, 5, 6, 7],
                                                    [1, 2, 31, 2, 2, 4, 28],
                                                    [1, 2, 31, 2, 2, 4, 28],
                                                ]).T,
                                                columns=["index", "OBJ1", "OBJ2", "OBJ3"],
                                            ),},
                {'id': 4, 'event_id': 575, 'datetime': datetime(2025, 5, 3, 2, 56, 9), 'record': 569, 'start_point': 680, 'end_point': 1600, 
                 'object_index': pd.DataFrame(
                                                np.array([
                                                    ['name_1', 'name_2', 'name_3', 'name_4', 'name_5', 'name_6', 'name_7'],
                                                    [1, 2, 3, 4, 5, 6, 7],
                                                    [1, 2, 31, 2, 2, 4, 28],
                                                    [1, 2, 31, 2, 2, 4, 28],
                                                ]).T,
                                                columns=["index", "OBJ1", "OBJ2", "OBJ3"],
                                            ),},
                {'id': 5, 'event_id': 576, 'datetime': datetime(2025, 5, 2, 15, 6, 37), 'record': 570, 'start_point': 790, 'end_point': 1760, 
                 'object_index': pd.DataFrame(
                                                np.array([
                                                    ['name_1', 'name_2', 'name_3', 'name_4', 'name_5', 'name_6', 'name_7'],
                                                    [1, 2, 3, 4, 5, 6, 7],
                                                    [1, 2, 31, 2, 2, 4, 28],
                                                    [1, 2, 31, 2, 2, 4, 28],
                                                ]).T,
                                                columns=["index", "OBJ1", "OBJ2", "OBJ3"],
                                            ),}
            ],

        }

        # Заглушечные данные для дат
        self.datetimedata = {
            2: [
                datetime(2025, 8, 25, 4, 32, 55).strftime("%Y-%m-%d %H:%M:%S"),
                datetime(2025, 8, 25, 15, 4, 23).strftime("%Y-%m-%d %H:%M:%S")
            ],
            5: [
                datetime(2025, 7, 10, 2, 22, 10).strftime("%Y-%m-%d %H:%M:%S"),
                datetime(2025, 7, 10, 8, 30, 45).strftime("%Y-%m-%d %H:%M:%S"),
                datetime(2025, 7, 10, 16, 55, 20).strftime("%Y-%m-%d %H:%M:%S")
            ],
            'default': [
                datetime(2024, 3, 16, 4, 32, 55).strftime("%Y-%m-%d %H:%M:%S"),
                datetime(2024, 4, 2, 15, 4, 23).strftime("%Y-%m-%d %H:%M:%S"),
                datetime(2024, 2, 10, 10, 15, 30).strftime("%Y-%m-%d %H:%M:%S")
            ]
        }

    # def get_example_df(self, pot_id=None):
    #     """Получить пример данных для ванны"""
    #     if pot_id and pot_id in self.exampledata:
    #         return self.exampledata[pot_id]
    #     return self.exampledata['default']

    def get_datetime_choices(self, pot_id=None):
        """Получить список дат для ванны"""
        if pot_id and pot_id in self.datetimedata:
            return self.datetimedata[pot_id]
        return self.datetimedata['default']

    def get_event_choices(self, pot_id=None):
        """Получить список событий для ванны (заглушка)"""
        # Здесь будет подключение к API
        base_choices = [231, 232, 233, 234, 235, 572, 573, 574, 575, 576]
        if pot_id == 2:
            return [231, 232, 233, 234, 235]
        elif pot_id == 5:
            return [572, 573, 574, 575, 576 ]
        return base_choices

    # Методы для подключения к реальному API (будут переопределены)
    def fetch_data_from_api(self, pot_id, params=None):
        """Метод для получения данных через API (заглушка)"""
        # Этот метод будет переопределен для реального API
        print(f"Запрос к API для ванны {pot_id} с параметрами {params}")
        # Здесь будет реальный код для обращения к API
        return self.get_example_df(pot_id)

    def fetch_datetime_from_api(self, pot_id, date_range=None):
        """Метод для получения дат через API (заглушка)"""
        # Этот метод будет переопределен для реального API
        print(f"Запрос к API для дат ванны {pot_id} с диапазоном {date_range}")
        # Здесь будет реальный код для обращения к API
        return self.get_datetime_choices(pot_id)

    def fetch_events_from_api(self, pot_id, date=None):
        """Метод для получения событий через API (заглушка)"""
        # Этот метод будет переопределен для реального API
        print(f"Запрос к API для событий ванны {pot_id} на дату {date}")
        # Здесь будет реальный код для обращения к API
        return self.get_event_choices(pot_id)
    

    def get_row_dataset_on_category(self, category=int, row_number=int):
        category_data = self.database_category[category]
        return category_data[row_number]
    


# if __name__ == "__main__":
#     app = DataProvider()
#     print('app.database_pot[2]', type(app.database_category[2])) # все строки для 2 ванны; type = list
#     category_data = app.database_category[2]
#     print("category_data[1]", type(category_data[1])) # полностью 2-я строка из списка данных по 2-й ванне; type = dict
#     print("category_data[1]['datetime']", type(category_data[1]['datetime'])) # просто время 2-й элемента в списке по 2-й ванне; type = datetime.datetime