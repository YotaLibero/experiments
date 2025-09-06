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
            101: [
                {'id': 1, 'event_id': 231, 'datetime': datetime(2025, 3, 1, 2, 34, 21), 'record': 566, 'start_point': 642, 'end_point': 1800,
                 'object_index': pd.DataFrame(
                                                np.array([
                                                    ['name_1', 'name_2', 'name_3', 'name_4', 'name_5', 'name_6', 'name_7'],
                                                    [1, 2, 3, 4, 5, 6, 7],
                                                    [1, 2, 31, 2, 2, 4, 28],
                                                    [1, 2, 31, 2, 2, 4, 28],
                                                ]).T,
                                                columns=["index", "OBJ1", "OBJ2", "OBJ3"],
                )},
                {'id': 2, 'event_id': 232, 'datetime': datetime(2025, 3, 1, 13, 21, 3), 'record': 567, 'start_point': 550, 'end_point': 2000, 
                 'object_index': pd.DataFrame(
                    np.array([
                        ['name_1', 'name_2', 'name_3', 'name_4', 'name_5', 'name_6', 'name_7'],
                        [1, 2, 3, 4, 5, 6, 7],
                        [1, 2, 31, 2, 2, 4, 28],
                        [1, 2, 31, 2, 2, 4, 28],
                    ]).T,
                    columns=["index", "OBJ1", "OBJ2", "OBJ3"],
                )},
                {'id': 3, 'event_id': 233, 'datetime': datetime(2025, 3, 1, 22, 17, 8), 'record': 568, 'start_point': 700, 'end_point': 1950, 
                 'object_index': pd.DataFrame(
                    np.array([
                        ['name_1', 'name_2', 'name_3', 'name_4', 'name_5', 'name_6', 'name_7'],
                        [1, 2, 3, 4, 5, 6, 7],
                        [1, 2, 31, 2, 2, 4, 28],
                        [1, 2, 31, 2, 2, 4, 28],
                    ]).T,
                    columns=["index", "OBJ1", "OBJ2", "OBJ3"],
                )},
                {'id': 4, 'event_id': 234, 'datetime': datetime(2025, 3, 3, 2, 56, 9), 'record': 569, 'start_point': 640, 'end_point': 1900, 
                 'object_index': pd.DataFrame(
                    np.array([
                        ['name_1', 'name_2', 'name_3', 'name_4', 'name_5', 'name_6', 'name_7'],
                        [1, 2, 3, 4, 5, 6, 7],
                        [1, 2, 31, 2, 2, 4, 28],
                        [1, 2, 31, 2, 2, 4, 28],
                    ]).T,
                    columns=["index", "OBJ1", "OBJ2", "OBJ3"],
                )},
                {'id': 5, 'event_id': 235, 'datetime': datetime(2025, 3, 3, 15, 6, 37), 'record': 570, 'start_point': 590, 'end_point': 1700, 
                 'object_index': pd.DataFrame(
                    np.array([
                        ['name_1', 'name_2', 'name_3', 'name_4', 'name_5', 'name_6', 'name_7'],
                        [1, 2, 3, 4, 5, 6, 7],
                        [1, 2, 31, 2, 2, 4, 28],
                        [1, 2, 31, 2, 2, 4, 28],
                    ]).T,
                    columns=["index", "OBJ1", "OBJ2", "OBJ3"],
                )}
            ],
            202: [
                {'id': 1, 'event_id': 572, 'datetime': datetime(2025, 5, 1, 2, 34, 21), 'record': 566, 'start_point': 645, 'end_point': 1870,
                 'object_index': pd.DataFrame(
                    np.array([
                        ['name_1', 'name_2', 'name_3', 'name_4', 'name_5', 'name_6', 'name_7'],
                        [1, 2, 3, 4, 5, 6, 7],
                        [1, 2, 31, 2, 2, 4, 28],
                        [1, 2, 31, 2, 2, 4, 28],
                    ]).T,
                    columns=["index", "OBJ1", "OBJ2", "OBJ3"],
                )},
                {'id': 2, 'event_id': 573, 'datetime': datetime(2025, 5, 1, 13, 21, 3), 'record': 567, 'start_point': 650, 'end_point': 2200, 
                 'object_index': pd.DataFrame(
                    np.array([
                        ['name_1', 'name_2', 'name_3', 'name_4', 'name_5', 'name_6', 'name_7'],
                        [1, 2, 3, 4, 5, 6, 7],
                        [1, 2, 31, 2, 2, 4, 28],
                        [1, 2, 31, 2, 2, 4, 28],
                    ]).T,
                    columns=["index", "OBJ1", "OBJ2", "OBJ3"],
                )},
                {'id': 3, 'event_id': 574, 'datetime': datetime(2025, 5, 1, 22, 17, 8), 'record': 568, 'start_point': 750, 'end_point': 2050, 
                 'object_index': pd.DataFrame(
                    np.array([
                        ['name_1', 'name_2', 'name_3', 'name_4', 'name_5', 'name_6', 'name_7'],
                        [1, 2, 3, 4, 5, 6, 7],
                        [1, 2, 31, 2, 2, 4, 28],
                        [1, 2, 31, 2, 2, 4, 28],
                    ]).T,
                    columns=["index", "OBJ1", "OBJ2", "OBJ3"],
                )},
                {'id': 4, 'event_id': 575, 'datetime': datetime(2025, 5, 2, 2, 56, 9), 'record': 569, 'start_point': 680, 'end_point': 1600, 
                 'object_index': pd.DataFrame(
                    np.array([
                        ['name_1', 'name_2', 'name_3', 'name_4', 'name_5', 'name_6', 'name_7'],
                        [1, 2, 3, 4, 5, 6, 7],
                        [1, 2, 31, 2, 2, 4, 28],
                        [1, 2, 31, 2, 2, 4, 28],
                    ]).T,
                    columns=["index", "OBJ1", "OBJ2", "OBJ3"],
                )},
                {'id': 5, 'event_id': 576, 'datetime': datetime(2025, 5, 2, 15, 6, 37), 'record': 570, 'start_point': 790, 'end_point': 1760, 
                 'object_index': pd.DataFrame(
                    np.array([
                        ['name_1', 'name_2', 'name_3', 'name_4', 'name_5', 'name_6', 'name_7'],
                        [1, 2, 3, 4, 5, 6, 7],
                        [1, 2, 31, 2, 2, 4, 28],
                        [1, 2, 31, 2, 2, 4, 28],
                    ]).T,
                    columns=["index", "OBJ1", "OBJ2", "OBJ3"],
                )}
            ],
        }

    def get_all_data_on_category(self, category=int):
        """Получить все данные по категории"""
        return self.database_category[category] if (int(category) == 101) or (int(category) == 202) else None

    def get_row_dataset_on_category(self, category=int, row_number=int):
        """Получить строку данных по категории и номеру"""
        try:
            category_data = self.get_all_data_on_category(category=int(category))
            return category_data[row_number] if 0 <= row_number < len(category_data) else None
        except Exception as exp:
            print(f'EXCEPTION: {exp}')
            return None

    def get_row_dataset_on_event_id(self, category, event):
        # print(f'❌ category {type(category)} - {category}')
        # print(f'❌ event {type(event)} - {event}')
        """Получить строку данных по категории и номеру"""
        try:
            category_data = self.get_all_data_on_category(category=int(category))
            row = [item for item in category_data 
                            if item['event_id'] == int(event)]
            return row[0]
        except Exception as exp:
            print(f'EXCEPTION: {exp}')
            return None

    def get_row_dataset_on_datetime_id(self, category, dt_id):
        # print(f'❌ category {type(category)} - {category}')
        # print(f'❌ dt_id {type(dt_id)} - {dt_id}')
        """Получить строку данных по категории и номеру"""
        # select_dt = datetime.strptime(dt_id, "%Y-%m-%d %H:%M:%S")
        try:
            category_data = self.get_all_data_on_category(category=int(category))
            row = [item for item in category_data 
                            if item['datetime'].strftime("%Y-%m-%d %H:%M:%S") == dt_id]
            # print(f'❌ row[0] {type(row[0])} - {row[0]}')
            return row[0]
        except Exception as exp:
            print(f'EXCEPTION: {exp}')
            return None

    def get_exist_last_row(self, category_id=None, select_dt_id=None):
        """Получить список дат для категории"""
        if select_dt_id is not None:
            category_data = self.get_all_data_on_category(category=int(category_id))
            dt_idxs = [item for item in category_data 
                            if item['datetime'].date().isoformat() == select_dt_id]
            dt_idxs = sorted(dt_idxs, key=lambda x: x['datetime'],  reverse=True)
        else:
            category_data = self.get_all_data_on_category(category=int(category_id))
            select_dt_id = sorted(category_data, key=lambda x: x['datetime'], reverse=True)
            select_dt_id = select_dt_id[0]
            select_dt_id = select_dt_id['datetime'].date().isoformat()
            dt_idxs = [item for item in category_data 
                            if item['datetime'].date().isoformat() == select_dt_id]
            dt_idxs = sorted(dt_idxs, key=lambda x: x['datetime'],  reverse=True)
        return dt_idxs[0] if len(dt_idxs) > 0 else None # list[str] or ['']

    def get_datetime_choices(self, category_id=None, select_dt_id=None):
        """Получить список дат для категории"""
        if (str(category_id) == '101') or (str(category_id) == '202'):
            category_data = self.get_all_data_on_category(category=int(category_id))
            dt_idxs = [item['datetime'].strftime("%Y-%m-%d %H:%M:%S") for item in category_data 
                            if item['datetime'].date().isoformat() == select_dt_id]
            return dt_idxs if len(dt_idxs) > 0 else [''] # list[str] or ['']
        else:
            return ['']

    def get_event_choices(self, category_id=None, select_dt_id=None, select_event_id=None):
        """Получить список событий для категории"""
        if (str(category_id) == '101') or (str(category_id) == '202'):
            category_data = self.get_all_data_on_category(category=int(category_id))
            if (select_dt_id is None) and (select_event_id is not None):
                dt_idxs = [str(item['event_id']) for item in category_data 
                                if int(item['event_id']) == select_event_id]
            else:
                dt_idxs = [str(item['event_id']) for item in category_data 
                                if item['datetime'].date().isoformat() == select_dt_id]
            return dt_idxs if len(dt_idxs) > 0 else ['']  # list[int] or ['']
        else:
            return ['']

    def get_van_records(self, category_id):
        """Получить записи категории из базы данных"""
        return self.database_category.get(category_id, [])

    def get_example_df(self, category_id=None):
        """Получить пример данных для категории"""
        if category_id and category_id in [101, 202]:
            # Возвращаем первый элемент из базы данных
            records = self.get_van_records(category_id)
            if records:
                return records[0]['object_index']
        return pd.DataFrame({
            'index': ['name_1', 'name_2', 'name_3'],
            'OBJ1': [1, 2, 3],
            'OBJ2': [1, 2, 31],
            'OBJ3': [1, 2, 31]
        })

    def get_param_values(self, category_id, event_id):
        """Получить значения start_point и end_point для события"""
        records = self.get_van_records(category_id)
        for record in records:
            if record['event_id'] == event_id:
                return {'start_point': record['start_point'], 'end_point': record['end_point']}
        return {'start_point': 2400, 'end_point': 2300}

    # Методы для подключения к реальному API (будут переопределены)
    def fetch_data_from_api(self, category_id, params=None):
        """Метод для получения данных через API (заглушка)"""
        print(f"Запрос к API для категории {category_id} с параметрами {params}")
        return self.get_example_df(category_id)

    def fetch_datetime_from_api(self, category_id, date_range=None):
        """Метод для получения дат через API (заглушка)"""
        print(f"Запрос к API для дат категории {category_id} с диапазоном {date_range}")
        return self.get_datetime_choices(category_id)

    def fetch_events_from_api(self, category_id, date=None):
        """Метод для получения событий через API (заглушка)"""
        print(f"Запрос к API для событий категории {category_id} на дату {date}")
        return self.get_event_choices(category_id)

# if _name_ == "__main__":
#     app = DataProvider()
#     print('app.database_category[101]', type(app.database_category[101])) # все строки для 101 категории; type = list
#     category_data = app.database_category[101]
#     print("category_data[1]", type(category_data[1])) # полностью 2-я строка из списка данных по 101-й категории; type = dict
#     print("category_data[1]['datetime']", type(category_data[1]['datetime'])) # просто время 2-й элемента в списке по 101-й категории; type = datetime.datetime