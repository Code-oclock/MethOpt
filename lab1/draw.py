import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

def draw(f, fileName, trace, z_max=None):
    x_vals, y_vals = trace[0], trace[1]
    z_vals = [f([x, y]) for x, y in zip(x_vals, y_vals)]
    
    # Создаем 3D поверхность функции
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])
    
    # Автоматическое определение пределов для куба
    z_axis_limit = [np.nanmin(Z), z_max] if z_max is not None else [np.nanmin(Z), np.nanmax(Z)]

    # Создаем объекты для визуализации
    surface = go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Rainbow',  # Более яркая палитра
        opacity=0.5,          # Меньшая прозрачность
        contours_z=dict(      # Контуры по Z
            show=True, 
            usecolormap=True,
            project_z=True
        ),
        name="Function Surface"
    )
    
    trajectory = go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='lines+markers',
        marker=dict(size=4, color='black'),  # Контрастный цвет
        line=dict(color='black', width=3),
        name="Optimization Path"
    )

    # Настройка макета
    layout = go.Layout(
        title='3D Gradient Descent Visualization',
        scene=dict(
            xaxis=dict(title='X', range=[-5, 5]),
            yaxis=dict(title='Y', range=[-5, 5]),
            zaxis=dict(title='Z', range=z_axis_limit),  # Ограничение по высоте
            aspectratio=dict(x=1, y=1, z=1 if z_max is None else 0.7),  # Пропорции куба
            camera=dict(eye=dict(x=1.3, y=0, z=1.5))
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    fig = go.Figure(data=[surface, trajectory], layout=layout)
    pio.write_html(fig, file=fileName, auto_open=False)