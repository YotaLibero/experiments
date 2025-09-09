from interface_app import InterfaceApp
import gradio as gr
from dataclasses import dataclass, field
from typing import Dict


# ------------------------------------------------------------------
# 7.  Запуск приложения
# ------------------------------------------------------------------
if __name__ == "__main__":
    # ---------------- State class ----------------
    @dataclass
    class StateInterface:
        # data: mapping object_id -> mapping slider_name -> value
        data: Dict[str, Dict[str, float]] = field(default_factory=dict)
        # current selected object id
        current: str = ""

    # ---------------- helpers ----------------
    def build_initial_state():
        objs = ["OBJ1", "OBJ2", "OBJ3"]
        default = {"slider_a": 50.0, "slider_b": 0.5}
        data = {o: default.copy() for o in objs}
        return StateInterface(data=data, current="OBJ1")

    # ---------------- Handlers ----------------
    def init_ui(state: StateInterface):
        """Инициализация интерфейса при загрузке — выставляем значения для текущего объекта."""
        vals = state.data.get(state.current, {"slider_a": 50.0, "slider_b": 0.5})
        return (
            gr.update(value=state.current),          # radio (на всякий случай)
            gr.update(value=vals["slider_a"]),       # slider A
            gr.update(value=vals["slider_b"]),       # slider B
            state
        )

    def on_obj_change(new_obj: str, slider_a_val: float, slider_b_val: float, state: StateInterface):
        """
        Вызывается, когда пользователь переключает объект.
        inputs: new_obj (новый выбор), текущие значения слайдеров (со старого объекта), state
        Логика:
        1) сохраняем значения в state для previous object (state.current)
        2) подгружаем значения для new_obj
        3) обновляем state.current
        4) возвращаем апдейты для обоих слайдеров и state
        """
        prev = state.current
        if prev:
            # сохраняем текущее состояние слайдеров для предыдущего объекта
            state.data.setdefault(prev, {})["slider_a"] = slider_a_val
            state.data.setdefault(prev, {})["slider_b"] = slider_b_val

        # убедимся, что для нового объекта есть запись (создаём с дефолтными значениями)
        if new_obj not in state.data:
            state.data[new_obj] = {"slider_a": 50.0, "slider_b": 0.5}

        new_vals = state.data[new_obj]
        state.current = new_obj

        # одним возвратом обновляем оба слайдера и state (атомарно)
        return (
            gr.update(value=new_vals["slider_a"]),
            gr.update(value=new_vals["slider_b"]),
            state
        )

    def on_sliders_change(slider_a_val: float, slider_b_val: float, obj_idx: str, state: StateInterface):
        """
        Обработчик для изменений любого из слайдеров.
        Обновляет state.data[current] = {slider_a, slider_b}.
        Мы привязываем этот обработчик к обоим слайдерам — входы включают оба слайдера,
        чтобы иметь полную пару значений.
        """
        # защищённо создаём запись, если вдруг её нет
        if obj_idx not in state.data:
            state.data[obj_idx] = {"slider_a": slider_a_val, "slider_b": slider_b_val}
        else:
            state.data[obj_idx]["slider_a"] = slider_a_val
            state.data[obj_idx]["slider_b"] = slider_b_val
        return state

    def show_state(state: StateInterface):
        """Просто выводим текущий state (для отладки)."""
        return str(state)

    # ---------------- UI ----------------
    with gr.Blocks() as demo:
        # сериализуем изменения стейта (предотвращаем одновременные гонки)
        demo.queue() # concurrency_count=1

        # создаём state (один объект на сессию)
        st = gr.State(build_initial_state())

        gr.Markdown("### Stateful sliders per object — пример (OBJ1/OBJ2/OBJ3)")

        obj_idx = gr.Radio(["OBJ1", "OBJ2", "OBJ3"], label="Object index", value="OBJ1")
        slider_a = gr.Slider(minimum=0, maximum=100, value=50, label="Slider A")
        slider_b = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="Slider B")
        show_btn = gr.Button("Показать сохранённый state (debug)")
        out = gr.Textbox(label="State (debug)")

        # Инициализация интерфейса при загрузке
        demo.load(init_ui, inputs=[st], outputs=[obj_idx, slider_a, slider_b, st])

        # Когда переключается объект:
        # Передаём в обработчик: новый выбор (obj_idx), текущие значения слайдеров (чтобы их сохранить),
        # и state. Возвращаем updates для sliders + state.
        obj_idx.change(
            on_obj_change,
            inputs=[obj_idx, slider_a, slider_b, st],
            outputs=[slider_a, slider_b, st]
        )

        # Когда изменяется любой слайдер — обновляем state для текущего объекта.
        # Мы передаём оба значения слайдеров + obj_idx + state, чтобы сохранить полную пару.
        slider_a.change(on_sliders_change, inputs=[slider_a, slider_b, obj_idx, st], outputs=[st])
        slider_b.change(on_sliders_change, inputs=[slider_a, slider_b, obj_idx, st], outputs=[st])

        # Кнопка для дебага — показать state
        show_btn.click(show_state, inputs=[st], outputs=[out])

    demo.launch()
