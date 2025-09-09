# state_interface.py

import pandas as pd

class StateInterface:
    def __init__(self):
        # текущее состояние
        self.obj_idx = None
        self.start_point = {}
        self.end_point = {}
        self.data_frames = {}

    def set_obj(self, obj_idx: str):
        """Установить выбранный объект"""
        self.obj_idx = obj_idx
        if obj_idx not in self.start_point:
            self.start_point[obj_idx] = 0
        if obj_idx not in self.end_point:
            self.end_point[obj_idx] = 100

    def update_points(self, obj_idx: str, start: int, end: int):
        """Обновить точки для выбранного объекта"""
        self.start_point[obj_idx] = start
        self.end_point[obj_idx] = end

    def attach_data(self, obj_idx: str, df: pd.DataFrame):
        """Сохраняем датафрейм для объекта"""
        self.data_frames[obj_idx] = df.copy()

    def get_filtered_data(self, obj_idx: str):
        """Вернуть датафрейм с учётом выбранных точек"""
        if obj_idx not in self.data_frames:
            return pd.DataFrame()

        df = self.data_frames[obj_idx]
        start = self.start_point.get(obj_idx, 0)
        end = self.end_point.get(obj_idx, len(df))
        return df.iloc[start:end]

    def get_state(self):
        """Текущее состояние"""
        return {
            "obj_idx": self.obj_idx,
            "start_point": self.start_point.get(self.obj_idx),
            "end_point": self.end_point.get(self.obj_idx),
        }
