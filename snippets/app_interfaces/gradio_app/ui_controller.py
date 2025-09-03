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
# 3.  Класс управления интерфейсом
# --------------------------------------------------------------
class UIController:
    def __init__(self, data_provider, updater):
        self.data_provider = data_provider
        self.updater = updater

    def greet(self, request: gr.Request):
        """Заполняет поле category_ID из query‑строки (если есть)."""
        query = dict(request.query_params)
        return int(query.get("name", 0))

    def edit_category_id(self, category):
        if int(category) == 2:
            record_id = 0
            calendar = datetime(2025, 8, 25)
        elif int(category) == 5:
            record_id = 1
            calendar = datetime(2025, 7, 10)
        else:
            record_id = -1
            calendar = datetime(1666, 1, 1)
        return category, calendar, self.get_updated_dropdown(category), record_id, self.get_updated_dropdown_dt(category)

    def category_s_change(self, category):
        """Обрабатывает изменение номера ванны."""
        if int(category) == 2:
            rec = 0
            dt = "2025-08-21"
        elif int(category) == 5:
            rec = 1
            dt = "2025-07-21"
        else:
            rec = -1
            dt = "2025-01-01"
        # После смены ванны сразу выводим актуальный список
        return category, rec, dt, self.get_updated_dropdown(category)

    def get_updated_dropdown(self, category, selected_choice=None) -> gr.update:
        """Возвращает актуальный набор вариантов для Dropdown."""
        choices = self.updater.get_choices()
        if selected_choice == None:
            value = choices[0] if choices else None
            print('нет значения для event_id')
        else:
            value = selected_choice
            print('есть значение для event_id')
        label = f"Ивенты для {category} ванны"
        return gr.update(choices=choices, value=value, label=label)

    def get_updated_dropdown_dt(self, category, selected_choice=None) -> gr.update:
        """Возвращает актуальный набор вариантов для Dropdown."""
        choices = self.updater.get_dt_choices()
        if selected_choice == None:
            value = choices[0] if choices else None
            print('нет значения для dt_id')
        else:
            value = selected_choice
            print('есть значение для dt_id')
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
        label = f"Ивенты для {category} ванны"
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
        try:
            # Проверяем, что calendar не None
            if calendar is None:
                print("Дата не выбрана")
                select_date_calendar = None
            else:
                select_date_calendar = datetime.fromisoformat(calendar.replace('Z', '+00:00'))
                select_date_calendar = select_date_calendar.date().isoformat()
        except (ValueError, AttributeError) as e:
            print(f"формат даты определён некорректно. Ожидалось 'ГГГГ-ММ-ДД'. Ошибка: {e}")
            select_date_calendar = None

        # Инициализация значений по умолчанию
        event_id = ''
        record_id = -1
        dt_id = ''

        if (int(category) == 2) or (int(category) == 5):
            category_data = self.data_provider.get_van_records(int(category))
            if category_data is None or len(category_data) == 0:
                print("Данные для ванны отсутствуют!")
                return dt_id, calendar.date().isoformat() if calendar else None, '', -1, -1

            # Если дата не выбрана, используем последнюю дату
            if select_date_calendar is None:
                if category_data:
                    select_date_calendar = datetime.strptime(category_data[-1]['datetime'], "%Y-%m-%d %H:%M:%S").date().isoformat()
                else:
                    print("Нет данных для определения последней даты")
                    return dt_id, calendar.date().isoformat() if calendar else None, '', -1, -1

            # Проверяем существование записи
            exists = any(datetime.strptime(item['datetime'], "%Y-%m-%d %H:%M:%S").date().isoformat() == select_date_calendar for item in category_data)
            if not exists:
                print(f"Записей на {select_date_calendar} не найдено")
                return dt_id, calendar.date().isoformat() if calendar else None, '', -1, -1
            else:
                # Находим все записи для выбранной даты и сортируем
                dt_idxs = [item for item in category_data 
                          if datetime.strptime(item['datetime'], "%Y-%m-%d %H:%M:%S").date().isoformat() == select_date_calendar]
                dt_idxs = sorted(dt_idxs, key=lambda x: datetime.strptime(x['datetime'], "%Y-%m-%d %H:%M:%S"), reverse=True)

                if dt_idxs:
                    dt_id = dt_idxs[0]  # Берем первую (самую позднюю) запись
                    event_id = dt_id['event_id']
                    record_id = dt_id['record']

                    return dt_id, calendar.date().isoformat() if calendar else None, \
                           self.get_updated_dropdown_dt(category, dt_id), \
                           self.get_updated_dropdown(category, event_id), \
                           record_id
                else:
                    return dt_id, calendar.date().isoformat() if calendar else None, '', -1, -1
        else:
            print("Такой ванны нет либо данные по существующей ванне отсутствуют!")
            return dt_id, calendar.date().isoformat() if calendar else None, '', -1, -1

    # def edit_event_id(self):
    #     pass
    #     return event, dt_id, record_id, obj_id, temp_0, solid, df_objs, plots

    # def edit_dt_id(self):
    #     pass
    #     return dt_id, event, record_id, obj_id, temp_0, solid, df_objs, plots

    # def edit_obj_idx(self):
    #     pass
    #     return obj_id, temp_0, solid, df_objs, plots

    # def edit_temp0(self):
    #     pass
    #     return temp_0, solid, df_objs, plots

    # def edit_solidus(self):
    #     pass
    #     return solid, temp_0, df_objs, plots
    