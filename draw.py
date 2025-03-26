import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.animation as animation

def draw(f, fileName, trace):
    x_vals, y_vals = trace[0], trace[1]
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=20, cmap
                ='viridis')
    plt.plot(x_vals, y_vals, marker='o', color='r', markersize=3, label="Path of gradient descent")
    plt.title("Gradient Descent with Constant Step Size")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.colorbar(label='Function Value')
    plt.grid(True)
    plt.savefig(fileName)
    plt.close()

def draw_interactive(f, fileName, trace):
    x_vals, y_vals = trace[0], trace[1]
    z_vals = [f([x, y]) for x, y in zip(x_vals, y_vals)]
    
    # Создаем 3D поверхность функции
    x = np.linspace(-4, 4, 50)
    y = np.linspace(-4, 4, 50)
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])
    
    # Создаем объекты для визуализации
    surface = go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        opacity=0.7,
        name="Function Surface"
    )
    
    trajectory = go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='lines+markers',
        marker=dict(size=4, color='red'),
        line=dict(color='red', width=2),
        name="Optimization Path"
    )
    
    # Настройка макета
    layout = go.Layout(
        title='3D Gradient Descent Visualization',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))  # Начальный угол обзора
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    fig = go.Figure(data=[surface, trajectory], layout=layout)
    pio.write_html(fig, file=fileName, auto_open=False)

def animate_2d_gradient_descent(f, path, x_opt, output_filename, title="Gradient Descent"):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Контурный график
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])
    ax.contour(X, Y, Z, levels=20, cmap='viridis')
    
    # Траектория
    line, = ax.plot([], [], 'ro-', markersize=3, linewidth=1, label='Path')
    start_point, = ax.plot([], [], 'bo', markersize=5, label='Start')
    end_point, = ax.plot([], [], 'g*', markersize=10, label='Minimum')
    
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    plt.grid(True)

    def init():
        line.set_data([], [])
        start_point.set_data(path[0][:1], path[1][:1])
        return line, start_point, end_point

    def update(frame):
        if frame == len(path[0]) - 1:  # Последний кадр
            end_point.set_data(x_opt[:1], x_opt[1:2])  # Показываем точку минимума
            end_point.set_visible(True)
        else:
            end_point.set_visible(False)
        line.set_data(path[0][:frame+1], path[1][:frame+1])
        ax.set_title(f"{title} (Iteration {frame+1})")
        return line, start_point, end_point

    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(path[0]),
        init_func=init, 
        blit=True, 
        interval=100
    )
    
    ani.save(output_filename, writer='pillow', fps=10)
    plt.close()
