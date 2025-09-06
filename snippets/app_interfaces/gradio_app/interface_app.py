import os
import gradio as gr
import pandas as pd
import numpy as np
import time
import threading
import random
from datetime import datetime, date
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
                category_id = gr.Dropdown(choices=['choose category', '101', '202'], label="Category")                     # номер категории
                calendar_input = gr.DateTime(
                                        label="Календарь",
                                        timezone="UTC",
                                        type="string", 
                                        value=datetime.now().strftime("%Y-%m-%d %H:%M:%S") #.date().isoformat() # type = str : строка формата ISO 8601 "%Y-%m-%d"
                                        )
                # Используем строковые представления дат для Dropdown
                datetime_id = gr.Dropdown(choices=[], label='DateTime') # type = str : строка формата ISO 8601 "%Y-%m-%d %H:%M:%S"
                event_id = gr.Dropdown(choices=[''], label="Event_ID")
                record_index = gr.Number(label="Record Index", )

            # ---------- Левая колонка ----------
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    obj_idx = gr.Radio(
                        ["OBJ1", "OBJ2", "OBJ3"], label="Object index", value="OBJ1"
                    )
                    start_point = gr.Slider(
                        minimum=200, maximum=4000, step=1, value=2400,
                        label="Start Point"
                    )
                    end_point = gr.Slider(
                        minimum=200, maximum=4000, step=1, value=2400,
                        label="End Point"
                    )
                    with gr.Row():
                        btn_prev = gr.Button("Prev")
                        btn_next = gr.Button("Next")
                    with gr.Row():
                        btn_save = gr.Button("Save to Database")
                    with gr.Row():
                        btn_find = gr.Button("Find Vector")
                    saved_lbl = gr.Label(value="Changes saved", color="green")
                    data_frame = gr.Dataframe(label="PeakParams")
                    graph_for_check = gr.Checkbox(value=False, label="Вопросы по графику")
                # ---------- Правая колонка ----------
                with gr.Column(scale=3, min_width=300):
                    # Заменяем Textbox на Plot для отображения графиков
                    plot_output = gr.Plot(label="Графики данных", format="png")  # Это теперь график
                    # plot_output = gr.Image(label="Графики данных", type="pil")

            # Привязываем события
            self.setupevent_handlers(demo, category_id, calendar_input, datetime_id, event_id, 
                                     obj_idx, start_point, end_point, data_frame,
                                     record_index, plot_output)

            return demo

    def setupevent_handlers(self, demo, category_id, calendar_input, datetime_id, event_id, 
                            obj_idx, start_point, end_point, data_frame,
                            record_index, plot_output):
        """Настройка обработчиков событий"""

        def update_with_pot(category):
            if (str(category) == '101') or (str(category) == '202'):
                row_values = self.data_provider.get_exist_last_row(category_id = int(category))
                print(type(row_values), row_values)
                calendar = row_values['datetime']
                if row_values is not None:
                    dt_value = row_values['datetime'].strftime("%Y-%m-%d %H:%M:%S")
                    record_id = int(row_values['record'])
                    event_id = str(row_values['event_id'])
                    datetime_choices = sorted(self.data_provider.get_datetime_choices(category_id=category, select_dt_id=calendar.date().isoformat()), reverse=True)
                    datetime_choices = datetime_choices if datetime_choices is not None else ['']
                    event_choices = sorted(self.data_provider.get_event_choices(category_id=category, select_dt_id=calendar.date().isoformat()), reverse=True)
                    event_choices = event_choices if event_choices is not None else ['']
                    return (
                        gr.update(choices=event_choices, value=event_id),
                        gr.update(choices=datetime_choices, value=dt_value),
                        record_id,
                        dt_value,
                        row_values['datetime'].strftime("%Y-%m-%d %H:%M:%S")
                    )
                else:
                    dt_value = ''
                    record_id = 0
                    event_id = ''
                    datetime_choices = ['']
                    event_choices = ['']

                    return (
                        gr.update(choices=event_choices, value=event_id),
                        gr.update(choices=datetime_choices, value=dt_value),
                        record_id,
                        dt_value,
                        calendar.strftime("%Y-%m-%d %H:%M:%S")
                    )

            if str(category) == 'choose category':
                record_id = ""
                dt_value = ""
                calendar = datetime.now().strftime("%Y-%m-%d %H:%M:%S") #datetime.now().date().isoformat()
                # Берем отсортированный по убыванию список дат
                datetime_choices = ['']
                event_choices = ['']
                return (
                    gr.update(choices=event_choices, value=event_choices[0] if event_choices else ''),
                    gr.update(choices=datetime_choices, value=datetime_choices[0] if datetime_choices else ''),
                    record_id,
                    dt_value,
                    calendar
                )

        # Обработчик изменения category_id - обновляет все компоненты с сортировкой по убыванию
        category_id.change(
            fn=update_with_pot,
            inputs=category_id,
            outputs=[event_id, datetime_id, record_index, calendar_input, calendar_input],  # calendar_input дважды для calendar
        )

        calendar_input.change(
            fn=self.ui_controller.edit_calendar,
            inputs=[category_id, calendar_input],
            outputs=[calendar_input, calendar_input, event_id, record_index, datetime_id],
        )

        # Обработчик изменения event_id - обновляет только базовые компоненты
        event_id.change(
            fn=self.ui_controller.edit_event_id,
            inputs=[category_id, event_id],
            outputs=[datetime_id, event_id, record_index, obj_idx, data_frame, start_point, end_point],  # calendar_input дважды для calendar        
            )
            # Обработчик изменения datetime_id - обновляет только базовые компоненты
        datetime_id.change(
            fn=self.ui_controller.edit_datetime_id,
            inputs=[category_id, datetime_id],
            outputs=[datetime_id, event_id, record_index, obj_idx, data_frame, start_point, end_point],  # calendar_input дважды для calendar        
            )

        # Обновление графика при изменении категории
        category_id.change(
            fn=PlotGenerator.create_plot,    #PlotGenerator.get_existing_image,  #PlotGenerator.create_plot,
            inputs=[category_id, obj_idx, start_point, end_point],
            outputs=plot_output
        )

        # Обновление графика при изменении номера замера
        event_id.change(
            fn=PlotGenerator.create_plot,    #PlotGenerator.get_existing_image,  #PlotGenerator.create_plot,
            inputs=[category_id, obj_idx, start_point, end_point],
            outputs=plot_output
        )

        # Обновление графика при изменении времени замера
        datetime_id.change(
            fn=PlotGenerator.create_plot,    #PlotGenerator.get_existing_image,  #PlotGenerator.create_plot,
            inputs=[category_id, obj_idx, start_point, end_point],
            outputs=plot_output
        )

        # Обновление графика при изменении параметров
        obj_idx.change(
            fn=PlotGenerator.create_plot,    #PlotGenerator.get_existing_image,  #PlotGenerator.create_plot,
            inputs=[category_id, obj_idx, start_point, end_point],
            outputs=plot_output
        )

        start_point.change(
            fn=PlotGenerator.create_plot,    #PlotGenerator.get_existing_image,  #PlotGenerator.create_plot,
            inputs=[category_id, obj_idx, start_point, end_point],
            outputs=plot_output
        )

        end_point.change(
            fn=PlotGenerator.create_plot,    #PlotGenerator.get_existing_image,  #PlotGenerator.create_plot,
            inputs=[category_id, obj_idx, start_point, end_point],
            outputs=plot_output
        )

        # Таймер, который каждые 10 сек обновляет список Dropdown
        # timer = gr.Timer(value=10)          # интервал в секундах
        # timer.tick(
        #         fn=self.ui_controller.refresh_dropdown,
        #         inputs=[category_id, event_id],          # получаем текущий выбор
        #         outputs=event_id,
        #     )

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