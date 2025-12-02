# pip install bokeh numpy pandas scikit-learn
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, DataTable, TableColumn, NumberFormatter, Div
from bokeh.layouts import row, column
import numpy as np
import pandas as pd
from math import sqrt

# fake data: true и predicted
np.random.seed(0)
x = np.arange(100)
y_true = np.sin(x / 10) + 0.1 * np.random.randn(100)
y_pred = np.sin(x / 10) + 0.2 * np.random.randn(100)  # «предсказание»

# посчитаем метрики
mae = float(np.mean(np.abs(y_true - y_pred)))
rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
# простой R2
ss_res = ((y_true - y_pred)**2).sum()
ss_tot = ((y_true - y_true.mean())**2).sum()
r2 = 1 - ss_res/ss_tot

metrics_df = pd.DataFrame({
    "metric": ["MAE", "RMSE", "R2"],
    "value": [mae, rmse, r2]
})

# Bokeh data sources
source_points = ColumnDataSource(dict(x=x, y_true=y_true, y_pred=y_pred))
source_metrics = ColumnDataSource(metrics_df)

# plot
p = figure(title="True vs Predicted", width=700, height=400, tools="pan,wheel_zoom,box_zoom,reset")
p.line('x', 'y_true', source=source_points, legend_label="true", line_width=2)
p.line('x', 'y_pred', source=source_points, legend_label="predicted", line_width=2, line_dash='dashed')
p.legend.location = "top_left"

# таблица метрик
columns = [
    TableColumn(field="metric", title="Metric"),
    TableColumn(field="value", title="Value", formatter=NumberFormatter(format="0.000"))
]
metrics_table = DataTable(source=source_metrics, columns=columns, width=250, height=140, index_position=None)

# текст пояснения
desc = Div(text="<b>Модель:</b> простой пример. Метрики вычислены для демонстрации.", width=250)

# компоновка: график слева, таблица+текст справа
layout = row(p, column(desc, metrics_table))

# export
output_file("plot_with_metrics.html", title="Plot + Metrics")
save(layout)
print("Сохранено в plot_with_metrics.html")





# # requirements: bokeh
# # pip install bokeh

# from bokeh.plotting import figure, output_file, save
# from bokeh.models import ColumnDataSource, HoverTool
# import numpy as np

# # данные
# x = np.linspace(0, 10, 200)
# y = np.sin(x) + 0.2 * np.random.randn(len(x))
# source = ColumnDataSource(dict(x=x, y=y))

# # фигура
# p = figure(title="Простой интерактивный граф", width=700, height=400, tools="pan,wheel_zoom,box_zoom,reset")
# p.circle('x', 'y', source=source, size=6, alpha=0.8)
# p.add_tools(HoverTool(tooltips=[("x", "@x{0.00}"), ("y", "@y{0.00}")]))  # подсказки

# # экспорт в standalone html
# output_file("simple_plot.html", title="Simple Bokeh Plot")
# save(p)  # создаст simple_plot.html
# print("Сохранено в simple_plot.html")
