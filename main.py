import numpy as np
import draw
import lib

if __name__ == "__main__":
    point = np.array([-3., -4.]) # Начальная точка для градиентного спуска

    # x_opt_fixed, path_fixed = lib.gradient_descent_fixed(lib.f, lib.grad_f, point.copy(), 0.999999)
    # print(path_fixed)
    # print(x_opt_fixed)
    # # draw.animate_2d_gradient_descent(path_fixed, x_opt_fixed, 'fixed_step.gif', "Fixed Step")
    # print("Completed")
    # draw.draw('fixed_step.png', path_fixed)
    # print("Completed")
    # draw.draw_interactive('fixed_step.html', path_fixed)
    # print("Completed")
    #
    # x_opt_decr, path_decr = lib.gradient_descent_decreasing(lib.f, lib.grad_f, point.copy(), h=0.99)
    # print(path_decr)
    # # draw.animate_2d_gradient_descent(path_decr, x_opt_decr, 'decreasing_step.gif', "Decreasing Step")
    # print("Completed")
    # draw.draw('decreasing_step.png', path_decr)
    # print("Completed")
    # draw.draw_interactive('decreasing_step.html', path_decr)
    # print("Completed")

    x_opt_armijo, path_armijo = lib.gradient_descent_armijo(lib.f, lib.grad_f, point.copy(), h=0.6)
    print(path_armijo)
    # draw.animate_2d_gradient_descent(path_armijo, x_opt_armijo, 'armijo_step.gif', "`Armijo` Step")
    print("Completed")
    draw.draw('armijo_step.png', path_armijo)
    print("Completed")
    draw.draw_interactive('armijo_step.html', path_armijo)
    print("Completed")
    #
    # x_opt_wolfe, path_wolfe = lib.gradient_descent_wolfe(lib.f, lib.grad_f, point.copy(), h=0.3)
    # print(path_armijo)
    # # draw.animate_2d_gradient_descent(path_wolfe, x_opt_wolfe, 'wolfe_step.gif', "`Wolfe` Step")
    # print("Completed")
    # draw.draw('wolfe_step.png', path_wolfe)
    # print("Completed")
    # draw.draw_interactive('wolfe_step.html', path_wolfe)
    # print("Completed")
    #
    # x_opt_golden, path_golden = lib.gradient_descent_golden(lib.f, lib.grad_f, point.copy())
    # print(path_golden)
    # # draw.animate_2d_gradient_descent(path_golden, x_opt_golden, 'golden_step.gif', "`Golden` Step")
    # print("Completed")
    # draw.draw('golden_step.png', path_golden)
    # print("Completed")
    # draw.draw_interactive('golden_step.html', path_golden)
    # print("Completed")
    #
    # x_opt_dichotomy, path_dichotomy = lib.gradient_descent_dichotomy(lib.f, lib.grad_f, point.copy())
    # print(path_dichotomy)
    # # draw.animate_2d_gradient_descent(path_dichotomy, x_opt_dichotomy, 'dichotomy_step.gif', "`Dichotomy` Step")
    # print("Completed")
    # draw.draw('dichotomy_step.png', path_dichotomy)
    # print("Completed")
    # draw.draw_interactive('dichotomy_step.html', path_dichotomy)
    # print("Completed")