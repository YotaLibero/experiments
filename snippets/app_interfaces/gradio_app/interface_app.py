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

from data_provider import DataProvider
from plot_generator import PlotGenerator
from auto_dropdown_updater import AutoDropdownUpdater
from ui_controller import UIController

# --------------------------------------------------------------
# 4.  Основной класс приложения
# --------------------------------------------------------------
class InterfaceApp:
    def __init__(self):
        # Создаем экземпляр провайдера данных
        self.data_provider = DataProvider()
        # Создаем экземпляр обновлятора
        self.updater = AutoDropdownUpdater(self.data_provider)
        # Создаем контроллер UI
        self.ui_controller = UIController(self.data_provider, self.updater)

        # Запускаем фоновый поток
        self.updater.start()


    def build_interface(self):
        """Создает интерфейс Gradio"""
        with gr.Blocks() as demo:
            # ---------- Верхняя строка ----------
            with gr.Row():
                category_id = gr.Dropdown(choices=[2, 5], value=2, label="Category_ID")                     # номер ванны
                calendar_input = gr.DateTime(
                                        label="Календарь",
                                        timezone="UTC",
                                        type="datetime", 
                                        value=datetime(2025, 9, 2)
                                        )
                # Используем строковые представления дат для Dropdown
                datetime_id = gr.Dropdown(
                    choices=self.data_provider.get_datetime_choices(2), 
                    value=self.data_provider.get_datetime_choices(2)[0], 
                    label="DateTime"
                ) 
                event_id = gr.Dropdown(choices=[523, 524], value=523, label="Event_ID")
                record_index = gr.Number(label="Record Index", value=3)

            # ---------- Левая колонка ----------
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    obj_idx = gr.Radio(
                        ["OBJ1", "OBJ2", "OBJ3"], label="Object index", value="OBJ1T"
                    )
                    start_point = gr.Slider(
                        minimum=200, maximum=4000, step=1, value=2400,
                        label="start point"
                    )
                    end_point = gr.Slider(
                        minimum=200, maximum=4000, step=1, value=2400,
                        label="end point"
                    )
                    with gr.Row():
                        btn_prev = gr.Button("Prev")
                        btn_next = gr.Button("Next")
                    with gr.Row():
                        btn_save = gr.Button("Save to Database")
                    with gr.Row():
                        btn_save = gr.Button("Find Vector")
                    saved_lbl = gr.Label(value="Changes saved", color="green")
                    data_frame = gr.Dataframe(self.data_provider.get_example_df(2), label="Description Charts")
                    graph_for_check = gr.Checkbox(value=False, label="Any questions?")
                # ---------- Правая колонка ----------
                with gr.Column(scale=3, min_width=300):
                    plot_output = gr.Plot(label="Графики данных", format="png")  # Это теперь график
                    # plot_output = gr.Image(label="Графики данных", type="pil")

            # Привязываем события
            self.setupevent_handlers(demo, category_id, calendar_input, datetime_id, event_id, 
                                     obj_idx, start_point, end_point, 
                                     record_index, plot_output)

            return demo

    def setupevent_handlers(self, demo, category_id, calendar_input, datetime_id, event_id, obj_idx, start_point, end_point, record_index, plot_output):
        """Настройка обработчиков событий"""

        # Функция для обновления базовых компонентов (взаимозаменяемые)
        def update_basic_components(category, event_id_value, datetime_id_value):
            """Обновляет базовые компоненты (event_id и datetime_id)"""
            # Получаем данные для выбранной ванны
            if int(category) == 2:
                record_id = 0
                dt_value = "2025-08-21"
                calendar = datetime(2025, 8, 25)
                # Обновляем список дат и событий
                datetime_choices = self.data_provider.get_datetime_choices(2)
                event_choices = self.data_provider.get_event_choices(2)
            elif int(category) == 5:
                record_id = 1
                dt_value = "2025-07-21"
                calendar = datetime(2025, 7, 10)
                # Обновляем список дат и событий
                datetime_choices = self.data_provider.get_datetime_choices(5)
                event_choices = self.data_provider.get_event_choices(5)
            else:
                record_id = -1
                dt_value = "1666-01-01"
                calendar = datetime(1666, 1, 1)
                # Обновляем список дат и событий
                datetime_choices = self.data_provider.get_datetime_choices()
                event_choices = self.data_provider.get_event_choices()

            # Обновляем компоненты
            return (
                gr.update(choices=event_choices, value=event_id_value if event_id_value in event_choices else event_choices[0] if event_choices else None),
                gr.update(choices=datetime_choices, value=datetime_id_value if datetime_id_value in datetime_choices else datetime_choices[0] if datetime_choices else None),
                record_id,
                dt_value,
                calendar
            )

        # Функция для обновления компонентов с учетом сортировки по убыванию
        def update_with_sorting(category):
            """Обновляет компоненты, беря первый элемент из отсортированного по убыванию списка"""
            # Получаем данные для выбранной ванны
            if int(category) == 2:
                record_id = 0
                dt_value = "2025-08-21 00:00:00"
                calendar = datetime(2025, 8, 25, 0, 0, 0)
                # Берем отсортированный по убыванию список дат
                datetime_choices = sorted(self.data_provider.get_datetime_choices(2), reverse=True)
                event_choices = sorted(self.data_provider.get_event_choices(2), reverse=True)
            elif int(category) == 5:
                record_id = 1
                dt_value = "2025-07-21 00:00:00"
                calendar = datetime(2025, 7, 21, 0, 0, 0)
                # Берем отсортированный по убыванию список дат
                datetime_choices = sorted(self.data_provider.get_datetime_choices(5), reverse=True)
                event_choices = sorted(self.data_provider.get_event_choices(5), reverse=True)
            else:
                record_id = -1
                dt_value = "1666-01-01 00:00:00"
                calendar = datetime(1666, 1, 1, 0, 0, 0)
                # Берем отсортированный по убыванию список дат
                datetime_choices = sorted(self.data_provider.get_datetime_choices(), reverse=True)
                event_choices = sorted(self.data_provider.get_event_choices(), reverse=True)

            # Обновляем компоненты
            return (
                gr.update(choices=event_choices, value=event_choices[0] if event_choices else None),
                gr.update(choices=datetime_choices, value=datetime_choices[0] if datetime_choices else None),
                record_id,
                dt_value,
                calendar
            )

        # Обработчик изменения category_id - обновляет все компоненты с сортировкой по убыванию
        category_id.change(
            fn=update_with_sorting,
            inputs=category_id,
            outputs=[event_id, datetime_id, record_index, calendar_input, calendar_input],  # calendar_input дважды для calendar
        )

        calendar_input.change(
            fn=self.ui_controller.edit_calendar,
            inputs=[category_id, calendar_input],
            outputs=[calendar_input, calendar_input, datetime_id, event_id, record_index], # calendar_input дважды для calendar
        )

        # Обработчик изменения event_id - обновляет только базовые компоненты
        event_id.change(
            fn=lambda category, event_val, datetime_val: update_basic_components(category, event_val, datetime_val),
            inputs=[category_id, event_id, datetime_id],
            outputs=[event_id, datetime_id, record_index, calendar_input, calendar_input],  # calendar_input дважды для calendar
        )

        # Обработчик изменения datetime_id - обновляет только базовые компоненты
        datetime_id.change(
            fn=lambda category, event_val, datetime_val: update_basic_components(category, event_val, datetime_val),
            inputs=[category_id, event_id, datetime_id],
            outputs=[event_id, datetime_id, record_index, calendar_input, calendar_input],  # calendar_input дважды для calendar
        )

        # Обновление графика при изменении ванны
        category_id.change(
            fn=PlotGenerator.create_plot,               # PlotGenerator.get_existing_image,  # PlotGenerator.create_plot,
            inputs=[category_id, obj_idx, start_point, end_point],
            outputs=plot_output
        )

        # Обновление графика при изменении параметров
        obj_idx.change(
            fn=PlotGenerator.get_existing_image,  #
            inputs=[category_id, obj_idx, start_point, end_point],
            outputs=plot_output
        )

        start_point.change(
            fn=PlotGenerator.create_plot,               # PlotGenerator.get_existing_image,  #PlotGenerator.create_plot,
            inputs=[category_id, obj_idx, start_point, end_point],
            outputs=plot_output
        )

        end_point.change(
            fn=PlotGenerator.create_plot,               # PlotGenerator.get_existing_image,  #PlotGenerator.create_plot,
            inputs=[category_id, obj_idx, start_point, end_point],
            outputs=plot_output
        )

        # Таймер, который каждые 10 сек обновляет список Dropdown
        timer = gr.Timer(value=10)          # интервал в секундах
        timer.tick(
                fn=self.ui_controller.refresh_dropdown,
                inputs=[category_id, event_id],          # получаем текущий выбор
                outputs=event_id,
            )

        # При выборе элемента выводим его в отдельный textbox
        # Используем отдельный компонент для вывода информации
        output_timer = gr.Textbox(label="Результат выбора")

        # Обработчик для отображения выбранного элемента
        def display_selection(choice, component_type):
            return f"Выбрано ({component_type}): {choice}"

        event_id.change(
                fn=lambda x: display_selection(x, "Event_ID"),
                inputs=event_id,
                outputs=output_timer,
            )

        datetime_id.change(
                fn=lambda x: display_selection(x, "DateTime"),
                inputs=datetime_id,
                outputs=output_timer,
            )

        calendar_input.change(
                fn=lambda x: f"Выбрано: {x}",
                inputs=calendar_input,
                outputs=output_timer,
            )
