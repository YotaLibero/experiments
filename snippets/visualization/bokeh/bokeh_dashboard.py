# bokeh_dashboard.py
import math
import numpy as np
import pandas as pd

from bokeh.models import (
    ColumnDataSource, DataTable, TableColumn, NumberFormatter,
    CheckboxGroup, Select, Slider, Button, Div, CustomJS,
    HoverTool, Toggle, Spacer
)
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import row, column, layout
from bokeh.io import curdoc
from bokeh.embed import file_html
from bokeh.resources import CDN

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------
# Генерация примера данных
# -----------------------
def generate_section_data(n_points=200, seed=0):
    """Возвращает dict с тремя series и x."""
    rng = np.random.RandomState(seed)
    x = np.arange(n_points)
    base = np.sin(x / 20.0)
    s1 = base + 0.1 * rng.randn(n_points)              # серия 1
    s2 = 0.8 * base + 0.15 * rng.randn(n_points) + 0.3 # серия 2 (смещение)
    s3 = np.cos(x / 10.0) * 0.5 + 0.1 * rng.randn(n_points)  # серия 3 (другая форма)
    return dict(x=x.tolist(), s1=s1.tolist(), s2=s2.tolist(), s3=s3.tolist())

# Создадим несколько "разделов"
SECTION_NAMES = ["Alpha", "Beta", "Gamma", "Delta"]
ALL_DATA = {name: generate_section_data(n_points=250, seed=ix*11) for ix, name in enumerate(SECTION_NAMES)}

# Для примера - считаем "true" как s1, "pred" мы сделаем смесь (для метрик)
def compute_metrics_for_section(data):
    x = np.array(data["x"])
    true = np.array(data["s1"])
    pred = 0.6 * np.array(data["s2"]) + 0.4 * np.array(data["s3"])
    mae = mean_absolute_error(true, pred)
    rmse = math.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    # топ-K ошибок
    diffs = np.abs(true - pred)
    topk_idx = np.argsort(-diffs)[:10]  # 10 largest errors
    topk_df = pd.DataFrame({
        "idx": topk_idx.tolist(),
        "x": x[topk_idx].tolist(),
        "true": true[topk_idx].tolist(),
        "pred": pred[topk_idx].tolist(),
        "abs_err": diffs[topk_idx].tolist()
    })
    metrics = {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}
    return metrics, topk_df

ALL_METRICS = {}
ALL_TOPK = {}
for name, d in ALL_DATA.items():
    m, t = compute_metrics_for_section(d)
    ALL_METRICS[name] = m
    ALL_TOPK[name] = t

# -----------------------
# UI и компоненты Bokeh
# -----------------------
def make_dashboard_doc(doc=None, use_server=False):
    """
    Создаёт layout. Если use_server==False, будет использовать CustomJS для всех взаимодействий
    и можно сохранить layout в standalone HTML. Если True, регистрирует Python callbacks в doc.
    """

    # --- DataSources ---
    # по умолчанию показываем первый раздел
    default_section = SECTION_NAMES[0]
    source = ColumnDataSource(ALL_DATA[default_section].copy())
    # отдельный источник для точек/hover (для linked selection)
    points_source = ColumnDataSource(dict(x=ALL_DATA[default_section]["x"],
                                         s1=ALL_DATA[default_section]["s1"],
                                         s2=ALL_DATA[default_section]["s2"],
                                         s3=ALL_DATA[default_section]["s3"]))

    # Источник для таблицы метрик (сводные метрики)
    metrics_df = pd.DataFrame({
        "metric": list(ALL_METRICS[default_section].keys()),
        "value": list(ALL_METRICS[default_section].values())
    })
    metrics_source = ColumnDataSource(metrics_df)

    # Источник для топ-K ошибок
    topk_source = ColumnDataSource(ALL_TOPK[default_section])

    # --- Controls ---
    select_section = Select(title="Раздел (Select):", value=default_section, options=SECTION_NAMES)
    slider_section = Slider(title="Раздел (Slider):", start=0, end=len(SECTION_NAMES)-1, value=0, step=1)
    checkbox_lines = CheckboxGroup(labels=["s1 (baseline)", "s2", "s3"], active=[0,1,2])
    toggle_grid = Toggle(label="Показывать сетку", active=True)
    btn_save_html = Button(label="Сохранить текущий вид в HTML (standalone)", button_type="success", width=260)
    # (Для standalone кнопка сработает через CustomJS, для server — через Python)

    # --- Plots ---
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
    p = figure(title=f"Section: {default_section} — несколько линий", width=800, height=380, tools=TOOLS)
    p.add_tools(HoverTool(tooltips=[("x","@x"),("s1","@s1{0.000}"),("s2","@s2{0.000}"),("s3","@s3{0.000}")]))
    r1 = p.line('x', 's1', source=source, line_width=2, legend_label="s1 (baseline)")
    r2 = p.line('x', 's2', source=source, line_width=2, legend_label="s2")
    r3 = p.line('x', 's3', source=source, line_width=2, legend_label="s3", line_dash="dashed")
    p.legend.click_policy = "hide"  # позволяет кликом на легенду скрывать линии

    # второй график: ошибки (true - pred)
    p_err = figure(title="Абсолютная ошибка (true - blended prediction)", width=800, height=240, tools=TOOLS, x_range=p.x_range)
    # вычислим blended pred on the fly in JS/Python; для простоты создадим колонку abs_err
    blended = 0.6 * np.array(ALL_DATA[default_section]["s2"]) + 0.4 * np.array(ALL_DATA[default_section]["s3"])
    abs_err = np.abs(np.array(ALL_DATA[default_section]["s1"]) - blended)
    err_source = ColumnDataSource(dict(x=ALL_DATA[default_section]["x"], abs_err=abs_err.tolist()))
    err_line = p_err.line('x', 'abs_err', source=err_source, line_width=2)

    # --- Tables ---
    columns_metrics = [
        TableColumn(field="metric", title="Metric"),
        TableColumn(field="value", title="Value", formatter=NumberFormatter(format="0.000"))
    ]
    table_metrics = DataTable(source=metrics_source, columns=columns_metrics, width=300, height=140, index_position=None)

    columns_topk = [
        TableColumn(field="idx", title="idx"),
        TableColumn(field="x", title="x"),
        TableColumn(field="true", title="true", formatter=NumberFormatter(format="0.000")),
        TableColumn(field="pred", title="pred", formatter=NumberFormatter(format="0.000")),
        TableColumn(field="abs_err", title="abs_err", formatter=NumberFormatter(format="0.000"))
    ]
    table_topk = DataTable(source=topk_source, columns=columns_topk, width=700, height=200)

    # --- Layout pieces and descriptions ---
    header = Div(text="<h1 style='margin:0px;'>Bokeh ML Dashboard — Full Example</h1>"
                      "<p style='margin:4px 0px 0px 0px;'>Покажи/скрой линии, переключай разделы (Select / Slider). "
                      "Таблица метрик и топ-K ошибок обновляются при выборе раздела.</p>",
                 width=1000)
    left_controls = column(select_section, slider_section, checkbox_lines, toggle_grid, btn_save_html, Spacer(height=10))
    right_panel = column(Div(text="<b>Сводные метрики</b>"), table_metrics, Div(text="<b>Top-K ошибок</b>"), table_topk)

    # main layout
    main = column(header, row(left_controls, column(p, p_err), right_panel))

    # ---------------------
    # Callbacks: JS (standalone) / Python (server)
    # ---------------------
    # Prepare data dictionaries for JS callback (must be JSON-serializable)
    all_data_js = {name: ALL_DATA[name] for name in SECTION_NAMES}
    all_metrics_js = {name: {"metric": list(ALL_METRICS[name].keys()), "value": list(ALL_METRICS[name].values())} for name in SECTION_NAMES}
    # topk: convert each dataframe to dict-of-lists
    all_topk_js = {name: ALL_TOPK[name].to_dict(orient='list') for name in SECTION_NAMES}

    if not use_server:
        # ---------- CustomJS callbacks for standalone HTML ----------
        # Select -> update everything
        select_callback = CustomJS(args=dict(source=source, metrics_source=metrics_source,
                                             topk_source=topk_source, all_data=all_data_js,
                                             all_metrics=all_metrics_js, all_topk=all_topk_js,
                                             p=p, err_source=err_source),
                                   code="""
            const section = cb_obj.value;
            // update main data
            const newd = all_data[section];
            source.data = {
                x: newd.x.slice(),
                s1: newd.s1.slice(),
                s2: newd.s2.slice(),
                s3: newd.s3.slice()
            };
            // update error source (compute blended prediction)
            const n = newd.x.length;
            const abs_err = [];
            for (let i=0;i<n;i++){
                const truev = newd.s1[i];
                const pred = 0.6 * newd.s2[i] + 0.4 * newd.s3[i];
                abs_err.push(Math.abs(truev - pred));
            }
            err_source.data = {x: newd.x.slice(), abs_err: abs_err};

            // update metrics
            metrics_source.data = {
                metric: all_metrics[section].metric.slice(),
                value: all_metrics[section].value.slice()
            };

            // update topk
            topk_source.data = {
                idx: all_topk[section].idx.slice(),
                x: all_topk[section].x.slice(),
                true: all_topk[section].true.slice(),
                pred: all_topk[section].pred.slice(),
                abs_err: all_topk[section].abs_err.slice()
            };

            // update title
            p.title.text = "Section: " + section + " — несколько линий";
            source.change.emit();
            err_source.change.emit();
            metrics_source.change.emit();
            topk_source.change.emit();
        """)
        select_section.js_on_change('value', select_callback)

        # Slider -> sync with select
        slider_callback = CustomJS(args=dict(select=select_section), code="""
            const idx = cb_obj.value;
            select.value = select.options[idx];
        """)
        slider_section.js_on_change('value', slider_callback)

        # when select changes, sync slider (so both update)
        select_to_slider = CustomJS(args=dict(slider=slider_section, options=SECTION_NAMES), code="""
            const sec = cb_obj.value;
            const idx = options.indexOf(sec);
            if (idx >= 0){
                slider.value = idx;
            }
        """)
        select_section.js_on_change('value', select_to_slider)

        # Checkbox -> toggle visibility of lines
        checkbox_callback = CustomJS(args=dict(r1=r1, r2=r2, r3=r3), code="""
            const active = cb_obj.active;
            r1.visible = active.includes(0);
            r2.visible = active.includes(1);
            r3.visible = active.includes(2);
        """)
        checkbox_lines.js_on_change('active', checkbox_callback)

        # Toggle grid
        grid_callback = CustomJS(args=dict(p=p), code="""
            const active = cb_obj.active;
            p.xgrid[0].visible = active;
            p.ygrid[0].visible = active;
        """)
        toggle_grid.js_on_change('active', grid_callback)

        # Save current view to HTML: create a file_html snapshot and prompt download by opening in new window
        # We use file_html server-side generation not available in pure JS; but we can create a data URL via client-side.
        # Simpler: provide user with a pre-generated export of the initial layout and message.
        btn_callback = CustomJS(code="""
            alert("В standalone режиме текущая страница уже является автономным HTML. Чтобы сохранить, используйте 'File -> Save Page As' в браузере.");
        """)
        btn_save_html.js_on_event('button_click', btn_callback)

    else:
        # ---------- Python callbacks for Bokeh server ----------
        def update_section(attr, old, new):
            # called when select_section changes value OR slider changes
            section = select_section.value
            # sync slider value
            idx = SECTION_NAMES.index(section)
            slider_section.value = idx

            # update data sources
            newd = ALL_DATA[section]
            source.data = dict(x=newd['x'], s1=newd['s1'], s2=newd['s2'], s3=newd['s3'])
            # update err_source
            blended = 0.6 * np.array(newd['s2']) + 0.4 * np.array(newd['s3'])
            abs_err = np.abs(np.array(newd['s1']) - blended)
            err_source.data = dict(x=newd['x'], abs_err=abs_err.tolist())

            # update metrics and topk
            m = ALL_METRICS[section]
            metrics_source.data = dict(metric=list(m.keys()), value=list(m.values()))
            t = ALL_TOPK[section]
            topk_source.data = t.to_dict(orient='list')

            p.title.text = f"Section: {section} — несколько линий"

        def slider_changed(attr, old, new):
            # update select when slider changes
            select_section.value = SECTION_NAMES[new]

        select_section.on_change('value', update_section)
        slider_section.on_change('value', slider_changed)

        def checkbox_py(attr, old, new):
            r1.visible = 0 in new
            r2.visible = 1 in new
            r3.visible = 2 in new

        checkbox_lines.on_change('active', checkbox_py)

        def toggle_grid_py(attr, old, new):
            p.xgrid[0].visible = toggle_grid.active
            p.ygrid[0].visible = toggle_grid.active

        toggle_grid.on_change('active', toggle_grid_py)

        def save_html_py():
            # save a snapshot of the current layout to a standalone HTML
            html = file_html(main, CDN, "Bokeh Dashboard Snapshot")
            out_fn = "bokeh_dashboard_full_snapshot.html"
            with open(out_fn, "w", encoding="utf-8") as f:
                f.write(html)
            # notify in console (can't easily pop client-side alert)
            print("Saved snapshot to", out_fn)

        btn_save_html.on_click(save_html_py)

    # register doc in server or return layout for saving
    if use_server:
        doc.add_root(main)
        doc.title = "Bokeh ML Dashboard (Server Mode)"
        return None
    else:
        return main

# -----------------------
# Entrypoints: standalone vs server
# -----------------------
def create_standalone_html():
    layout_obj = make_dashboard_doc(use_server=False)
    output_file("bokeh_dashboard_full.html", title="Bokeh Dashboard Full Example")
    save(layout_obj)
    print("Standalone HTML сгенерирован: bokeh_dashboard_full.html")

if __name__.startswith("bokeh"):
    # If running under bokeh serve, `__name__` begins with 'bokeh'
    make_dashboard_doc(doc=curdoc(), use_server=True)
else:
    # When run directly, produce standalone html
    create_standalone_html()
