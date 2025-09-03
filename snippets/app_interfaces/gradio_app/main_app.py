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

from interface_app import InterfaceApp

# ------------------------------------------------------------------
# 7.  Запуск приложения
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Можно использовать заглушку или реальный API
    # app = InterfaceApp()  # Использует заглушку

    # Или можно использовать реальный API (демонстрация)
    # api_provider = RealAPIProvider("https://api.example.com", "your_api_key")
    # app = InterfaceAppWithAPI(api_provider)  # Нужно создать дополнительный класс

    app = InterfaceApp()
    demo = app.build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)