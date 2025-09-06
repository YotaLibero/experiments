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

class UIController:
    def __init__(self, data_provider, updater):
        self.data_provider = data_provider
        self.updater = updater

    def greet(self, request: gr.Request):
        """Заполняет поле Pot_ID из query‑строки (если есть)."""
        query = dict(request.query_params)
        return int(query.get("name", 0))

    def get_updated_dropdown(self, category, curr_choice=None) -> gr.update:
        """Возвращает актуальный набор вариантов для Dropdown."""
        choices = self.updater.get_choices()
        if curr_choice is None:
            value = choices[0] if choices else None
        else:
            value = curr_choice
        label = f"Ивенты для {category} категории"
        return gr.update(choices=choices, value=value, label=label)

    def get_updated_dropdown_dt(self, category, curr_choice=None) -> gr.update:
        """Возвращает актуальный набор вариантов для Dropdown."""
        choices = self.updater.get_dt_choices()
        if curr_choice is None:
            value = choices[0] if choices else None
        else:
            value = curr_choice
        label = f"Datetime for {category} category"
        return gr.update(choices=choices, value=value, label=label)

    def refresh_dropdown(self, category, current_choice):
        """
        Вызывается таймером каждые 10 сек.
        Сохраняет текущий выбранный элемент, если он всё ещё присутствует в новом списке.
        """
        choices = self.updater.get_choices()
        # Если текущий выбор есть в новых вариантах — оставляем его,
        # иначе выбираем первый элемент.
        if current_choice in choices:
            value = current_choice
        else:
            value = choices[0] if choices else None
        label = f"Ивенты для {category} категории"
        return gr.update(choices=choices, value=value, label=label)

    def show_selected(self, choice):
        """Выводит в отдельный textbox выбранный элемент (не меняет dropdown)."""
        return f"Выбрано: {choice}"

    def extract_date(self, datetime_value):
        if datetime_value is None:
            return None
        # Преобразуем строку в объект datetime, если нужно
        if isinstance(datetime_value, str):
            dt = datetime.fromisoformat(datetime_value)
        else:
            dt = datetime_value
        # Возвращаем только дату
        return dt.date()

    def edit_calendar(self, category, calendar): # category=str и calendar=str
        print("✅ calendar:", calendar)
        try:
            # Проверяем, что calendar не None
            if calendar is None:
                print("Дата не выбрана")
                select_date_calendar = None
            else:
                # Обрабатываем разные форматы даты
                if isinstance(calendar, str):
                    # Если строка - пытаемся парсить
                    if 'T' in calendar:
                        select_date_calendar = datetime.strptime(calendar, "%Y-%m-%d %H:%M:%S") # calendar.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        select_date_calendar = datetime.strptime(calendar, "%Y-%m-%d %H:%M:%S")
                else:
                    select_date_calendar = calendar

                # Получаем только дату
                select_date_calendar = select_date_calendar.date().isoformat()
        except (ValueError, AttributeError) as e:
            print(f"формат даты определён некорректно. Ожидалось 'ГГГГ-ММ-ДД'. Ошибка: {e}")
            select_date_calendar = None

        # Инициализация значений по умолчанию
        event_id = ''
        record_id = 0
        dt_id = ''
        print("✅ select_date_calendar:", select_date_calendar)

        if (int(category) == 101) or (int(category) == 202):
            category_data = self.data_provider.get_all_data_on_category(category=int(category))
            if category_data is None or len(category_data) == 0:
                print("🛑 Данные для категории отсутствуют!")
                # Возвращаем только calendar и dt_id, остальные оставляем как есть
                return dt_id, calendar, '', 0, ''

            # Если дата не выбрана, используем последнюю дату
            if select_date_calendar is None:
                if category_data:
                    select_date_calendar = category_data[-1]['datetime'].date().isoformat()
                else:
                    print("🛑 Нет данных для определения последней даты")
                    return dt_id, calendar, '', 0, ''

            # Проверяем существование записи
            exists = self.data_provider.get_exist_last_row(category_id=int(category), select_dt_id=select_date_calendar)
            if not exists:
                print(f"🛑 Записей на {select_date_calendar} не найдено")
                return dt_id, calendar, '', 0, ''
            else:
                # Находим все записи для выбранной даты и сортируем
                dt_id = self.data_provider.get_exist_last_row(category_id=int(category), select_dt_id=select_date_calendar)
                event_id = str(dt_id['event_id'])
                record_id = int(dt_id['record'])

                # Обновляем datetime_id для соответствующей дате
                datetime_choices = self.data_provider.get_datetime_choices(category_id=int(category), select_dt_id=dt_id['datetime'].date().isoformat())  # Исправлено: убран select_dt_id
                datetime_id_update = gr.update(choices=datetime_choices, value=dt_id['datetime'].strftime("%Y-%m-%d %H:%M:%S"))

                # Обновляем event_id для соответствующей дате
                event_choices = self.data_provider.get_event_choices(category_id=int(category), select_dt_id=dt_id['datetime'].date().isoformat())  # Исправлено: убран select_dt_id
                event_id_update = gr.update(choices=event_choices, value=event_id)

                # Возвращаем только нужные компоненты
                return dt_id['datetime'].strftime("%Y-%m-%d %H:%M:%S"), calendar, event_id_update, gr.update(value=record_id), datetime_id_update
        else:
            print("🛑 Такой категории нет либо данные по существующей категории отсутствуют!")
            return '', calendar, '', 0, ''

    def edit_event_id(self, category, event_id):
        """Обновление выбранного event_id и связанных параметров"""
        if (str(category) == '101') or(str(category) == '202'):
            # Находим запись по event_id
            record = self.data_provider.get_row_dataset_on_event_id(category=int(category), event=int(event_id))
            if record:
                # Получаем параметры для события
                start_point = record['start_point']
                end_point = record['end_point']

                # Получаем DataFrame для объекта
                df_objs = record['object_index']

                # Обновляем все значения
                return record['datetime'].strftime("%Y-%m-%d %H:%M:%S"), str(event_id), record['record'], 'OBJ1', df_objs, start_point, end_point
            else:
                # Если запись не найдена, возвращаем значения по умолчанию
                return "", str(event_id), -1, 'OBJ1', None, 2400, 2300
        else:
            # Если запись не найдена, возвращаем значения по умолчанию
            return "", str(event_id), -1, 'OBJ1', None, 2400, 2300 # ['OBJ1', 'OBJ2', 'OBJ3']

    def edit_datetime_id(self, category, dt_id):
        """Обновление выбранного datetime_id и связанных параметров"""
        if (str(category) == '101') or(str(category) == '202'):
            # Находим запись по event_id
            record = self.data_provider.get_row_dataset_on_datetime_id(category=int(category), dt_id=dt_id)
            if record:
                # Получаем параметры для события
                start_point = record['start_point']
                end_point = record['end_point']

                # Получаем DataFrame для объекта
                df_objs = record['object_index']

                # Обновляем все значения
                return record['datetime'].strftime("%Y-%m-%d %H:%M:%S"), str(record['event_id']), record['record'], 'OBJ1', df_objs, start_point, end_point
            else:
                # Если запись не найдена, возвращаем значения по умолчанию
                return dt_id, '', -1, 'OBJ1', None, 2400, 2300
        else:
            # Если запись не найдена, возвращаем значения по умолчанию
            return dt_id, '', -1, 'OBJ1', None, 2400, 2300 # ['OBJ1', 'OBJ2', 'OBJ3']

    def edit_obj_idx(self, category, obj_idx, event_id):
        """Обновление выбранного obj_idx и связанных параметров"""
        # Находим запись по event_id
        category_data = self.data_provider.get_van_records(int(category))
        record = None
        for item in category_data:
            if item['event_id'] == event_id:
                record = item
                break

        if record:
            # Получаем DataFrame для объекта
            df_objs = record['object_index']

            # Обновляем все значения
            return obj_idx, df_objs, None
        else:
            # Если запись не найдена, возвращаем значения по умолчанию
            return obj_idx, self.data_provider.get_example_df(category), None

    def edit_start_point(self, category, start_point, event_id):
        """Обновление значения start_point"""
        # Находим запись по event_id
        category_data = self.data_provider.get_van_records(int(category))
        record = None
        for item in category_data:
            if item['event_id'] == event_id:
                record = item
                break

        if record:
            # Обновляем значение start_point
            return start_point, None
        else:
            # Если запись не найдена, возвращаем значение по умолчанию
            return 2400, None

    def edit_end_point(self, category, end_point, event_id):
        """Обновление значения end_point"""
        # Находим запись по event_id
        category_data = self.data_provider.get_van_records(int(category))
        record = None
        for item in category_data:
            if item['event_id'] == event_id:
                record = item
                break

        if record:
            # Обновляем значение end_point
            return end_point, None
        else:
            # Если запись не найдена, возвращаем значение по умолчанию
            return 2300, None