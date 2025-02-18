
import tkinter as tk
import tkinter.messagebox as messagebox
import tkinter.simpledialog as simpledialog
from email.policy import default
from tkinter import ttk, StringVar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

from lab_01.functions import SimulationResults, run_simulation_once, run_simulation_passive
from lab_01.request import REQUEST_TYPE_ONE, REQUEST_TYPE_TWO


class UI:
    def __init__(self, master):
        self.master = master
        master.title("Лабораторная работа №1. ИУ7-83Б Авдейкина")
        master.geometry("500x700")  # Фиксированный размер окна

        self.font = ("Courier", 14)  # Определяем шрифт
        self.padding = 5  # Отступ внутри LabelFrame
        self.label_width = 10  # Ширина меток
        self.experiment_type = tk.StringVar(value="passive")  # Default to passive

        self.DESCRIPTION = ("\tПрограмма позволяет выполнить моделирование СМО с абсолютными приоритетами, "
                            "в которой\nсодержатся два генератора, бесконечный буфер и обслуживающий аппарат.\n"
                            "\tВ системе имеются заявки двух типов, каждый тип генерируются соответствующим генератором.\n"
                            "\tЗаявки первого типа приоритетнее заявок второго типа.\n"
                            "\tИнтенсивность заявок первого типа - lambda1.\n"
                            "\tИнтенсивность заявок второго типа - lambda2.\n"
                            "\tИнтенсивность обслуживания заявок первого типа - mu1.\n"
                            "\tИнтенсивность обслуживания заявок второго типа - mu2.")

        self.create_widgets()

    def create_widgets(self):
        # Строка меню
        self.menubar = tk.Menu(self.master)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="Инфо", command=self.open_info_window)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Выход", command=self.master.quit)
        self.menubar.add_cascade(label="Меню", menu=self.filemenu)
        self.master.config(menu=self.menubar)

        # Группа 1: Генератор 1
        self.generator1_frame = tk.LabelFrame(self.master, text="Генератор 1", font=self.font, padx=self.padding, pady=self.padding)
        self.generator1_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        self.lambda1_label_val = StringVar()
        self.lambda1_label_val.set("lambda1\n(min, max)")
        self.lambda1_label = tk.Label(self.generator1_frame, textvariable=self.lambda1_label_val, font=self.font, anchor='w', justify='left')
        self.lambda1_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.lambda1_spinbox_min_val = tk.DoubleVar()
        self.lambda1_spinbox_min_val.set(1.0)
        self.lambda1_spinbox_min = tk.Spinbox(self.generator1_frame, from_=0.1, to=1000, increment=0.1, font=self.font, textvariable=self.lambda1_spinbox_min_val)
        self.lambda1_spinbox_min.grid(row=0, column=1, padx=5, pady=5, sticky="e")
        self.lambda1_spinbox_max_val = tk.DoubleVar()
        self.lambda1_spinbox_max_val.set(5.0)
        self.lambda1_spinbox_max = tk.Spinbox(self.generator1_frame, from_=0.1, to=1000, increment=0.1, font=self.font, textvariable=self.lambda1_spinbox_max_val)
        self.lambda1_spinbox_max.grid(row=0, column=2, padx=5, pady=5, sticky="e")


        # Группа 2: Генератор 2
        self.generator2_frame = tk.LabelFrame(self.master, text="Генератор 2", font=self.font, padx=self.padding, pady=self.padding)
        self.generator2_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        self.lambda2_label_val = StringVar()
        self.lambda2_label_val.set("lambda2\n(min, max)")
        self.lambda2_label = tk.Label(self.generator2_frame, textvariable=self.lambda2_label_val, font=self.font, anchor='w', justify='left')
        self.lambda2_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.lambda2_spinbox_min_val = tk.DoubleVar()
        self.lambda2_spinbox_min_val.set(1.0)
        self.lambda2_spinbox_min = tk.Spinbox(self.generator2_frame, from_=0.1, to=1000, increment=0.1, font=self.font, textvariable=self.lambda2_spinbox_min_val)
        self.lambda2_spinbox_min.grid(row=0, column=1, padx=5, pady=5, sticky="e")
        self.lambda2_spinbox_max_val = tk.DoubleVar()
        self.lambda2_spinbox_max_val.set(5.0)
        self.lambda2_spinbox_max = tk.Spinbox(self.generator2_frame, from_=0.1, to=1000, increment=0.1, font=self.font, textvariable=self.lambda2_spinbox_max_val)
        self.lambda2_spinbox_max.grid(row=0, column=2, padx=5, pady=5, sticky="e")

        # Группа 3: Обслуживающий аппарат
        self.service_frame = tk.LabelFrame(self.master, text="Обслуживающий аппарат", font=self.font, padx=self.padding, pady=self.padding)
        self.service_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        # Строка 1: mu1
        self.mu1_label_val = StringVar()
        self.mu1_label_val.set("mu1\n(min, max)")
        self.mu1_label = tk.Label(self.service_frame, textvariable=self.mu1_label_val, font=self.font, anchor='w', justify='left')
        self.mu1_label.grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.mu1_spinbox_min_val = tk.DoubleVar()
        self.mu1_spinbox_min_val.set(1.0)
        self.mu1_spinbox_min = tk.Spinbox(self.service_frame, from_=0.1, to=1000, increment=0.1, font=self.font, textvariable=self.mu1_spinbox_min_val)
        self.mu1_spinbox_min.grid(row=0, column=1, padx=5, pady=2, sticky="e")
        self.mu1_spinbox_max_val = tk.DoubleVar()
        self.mu1_spinbox_max_val.set(5.0)
        self.mu1_spinbox_max = tk.Spinbox(self.service_frame, from_=0.1, to=1000, increment=0.1, font=self.font, textvariable=self.mu1_spinbox_max_val)
        self.mu1_spinbox_max.grid(row=0, column=2, padx=5, pady=2, sticky="e")

        # Строка 2: mu2
        self.mu2_label_val = StringVar()
        self.mu2_label_val.set("mu2\n(min, max)")
        self.mu2_label = tk.Label(self.service_frame, textvariable=self.mu2_label_val, font=self.font, anchor='w', justify='left')
        self.mu2_label.grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.mu2_spinbox_min_val = tk.DoubleVar()
        self.mu2_spinbox_min_val.set(1.0)
        self.mu2_spinbox_min = tk.Spinbox(self.service_frame, from_=0.1, to=1000, increment=0.1, font=self.font, textvariable=self.mu2_spinbox_min_val)
        self.mu2_spinbox_min.grid(row=1, column=1, padx=5, pady=2, sticky="e")
        self.mu2_spinbox_max_val = tk.DoubleVar()
        self.mu2_spinbox_max_val.set(5.0)
        self.mu2_spinbox_max = tk.Spinbox(self.service_frame, from_=0.1, to=1000, increment=0.1, font=self.font, textvariable=self.mu2_spinbox_max_val)
        self.mu2_spinbox_max.grid(row=1, column=2, padx=5, pady=2, sticky="e")

        # Группа 4: Моделирование
        self.simulation_frame = tk.LabelFrame(self.master, text="Моделирование", font=self.font, padx=self.padding, pady=self.padding)
        self.simulation_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        # Строка 1: Количество заявок
        self.requests_label = tk.Label(self.simulation_frame, text="Количество\nзаявок", font=self.font,
                                       width=self.label_width, anchor='w', justify='left')
        self.requests_label.grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.requests_spinbox_val = tk.IntVar()
        self.requests_spinbox_val.set(300)
        self.requests_spinbox = tk.Spinbox(self.simulation_frame, from_=1, to=1000, increment=10, font=self.font, textvariable=self.requests_spinbox_val)
        self.requests_spinbox.grid(row=0, column=1, columnspan=2, sticky='e', padx=5, pady=5)

        # Строка 2: Кнопка "Запуск"
        self.run_button = tk.Button(self.simulation_frame, text="Запуск", command=self.run_simulation, font=self.font)
        self.run_button.grid(row=1, column=0, columnspan=3, sticky='ew', pady=5, padx=5)

        self.result_frame = tk.LabelFrame(self.master, text="Результаты", font=self.font, padx=self.padding, pady=self.padding)
        self.result_frame.grid(row=4, column=0, padx=10, pady=5, sticky="ew")
        self.result_frame_label_val = StringVar()
        self.result_frame_label_val.set("Тут будут результаты вычислений")
        self.result_frame_label = tk.Label(self.result_frame, textvariable=self.result_frame_label_val, font=self.font, anchor='w', justify='left')
        self.result_frame_label.grid(row=0, column=0, padx=5, pady=2, sticky="w")

        # Radio buttons for experiment type
        self.radio_frame = tk.Frame(self.master)
        self.radio_frame.grid(row=5, column=0, pady=5)  # Place at the bottom

        self.one_point_radio = tk.Radiobutton(self.radio_frame, text="Одна точка", font=self.font,
                                              variable=self.experiment_type, value="one_point",
                                              command=self.update_ui)
        self.one_point_radio.grid(row=0, column=0, padx=10)

        self.passive_radio = tk.Radiobutton(self.radio_frame, text="Пассивный эксперимент", font=self.font,
                                            variable=self.experiment_type, value="passive",
                                            command=self.update_ui)
        self.passive_radio.grid(row=0, column=1, padx=10)

        # Ensure 'Пассивный эксперимент' is selected by default
        self.one_point_radio.select()
        self.update_ui()

        # Настройка весов столбцов и строк
        self.master.columnconfigure(0, weight=1)
        self.generator1_frame.columnconfigure(1, weight=1)
        self.generator2_frame.columnconfigure(1, weight=1)
        self.service_frame.columnconfigure(1, weight=1)
        self.simulation_frame.columnconfigure(1, weight=1)
        self.generator1_frame.columnconfigure(2, weight=1)
        self.generator2_frame.columnconfigure(2, weight=1)
        self.service_frame.columnconfigure(2, weight=1)
        self.simulation_frame.columnconfigure(2, weight=1)

        for i in range(6):
            self.master.rowconfigure(i, weight=1)

    def update_ui(self):
        experiment_mode = self.experiment_type.get()

        # Toggle visibility of min/max labels and spinboxes based on experiment type
        lambda1_widgets = [self.lambda1_spinbox_max]
        lambda2_widgets = [self.lambda2_spinbox_max]
        mu1_widgets = [self.mu1_spinbox_max]
        mu2_widgets = [self.mu2_spinbox_max]

        all_widgets = lambda1_widgets + lambda2_widgets + mu1_widgets + mu2_widgets

        if experiment_mode == "one_point":
            for widget in all_widgets:
                widget.grid_remove()
                self.result_frame.grid()
                self.lambda1_label_val.set("lambda1")
                self.lambda2_label_val.set("lambda2")
                self.mu1_label_val.set("mu1")
                self.mu2_label_val.set("mu2")
        else:  # passive
            for widget in all_widgets:
                self.result_frame.grid_remove()
                self.lambda1_label_val.set("lambda1\n(min, max)")
                self.lambda2_label_val.set("lambda2\n(min, max)")
                self.mu1_label_val.set("mu1\n(min, max)")
                self.mu2_label_val.set("mu2\n(min, max)")
                widget.grid()

        self.generator1_frame.columnconfigure(2, weight=1)
        self.generator2_frame.columnconfigure(2, weight=1)
        self.service_frame.columnconfigure(2, weight=1)
        self.simulation_frame.columnconfigure(2, weight=1)

    def open_info_window(self):
        info_window = tk.Toplevel(self.master)
        info_window.title("Описание системы")

        label = tk.Label(info_window, text=self.DESCRIPTION, anchor="center", font=self.font, justify='left', width=100)
        label.grid(row=0, column=0, padx=20, pady=20, sticky="nsew") # Использован grid
        info_window.rowconfigure(0, weight=1)
        info_window.columnconfigure(0, weight=1)


    def run_simulation(self):
        experiment_mode = self.experiment_type.get()

        if experiment_mode == "one_point":
            requests_processed = self.requests_spinbox_val.get()
            requests_per_generator = requests_processed * 2000
            lambda1 = self.lambda1_spinbox_min_val.get()
            lambda2 = self.lambda2_spinbox_min_val.get()
            mu1 = self.mu1_spinbox_min_val.get()
            mu2 = self.mu2_spinbox_min_val.get()
            results: SimulationResults = run_simulation_once(
                requests_per_generator,
                requests_processed,
                lambda1, lambda2, mu1, mu2
            )

            # result_load_ratio_1 = results.stats.proc_by_type[REQUEST_TYPE_ONE] / results.stats.avg_wait_times[REQUEST_TYPE_ONE]
            # result_load_ratio_2 = ((results.stats.proc_by_type[REQUEST_TYPE_TWO] + results.stats.proc_by_type[REQUEST_TYPE_ONE])
            #                        / (results.stats.avg_wait_times[REQUEST_TYPE_TWO] + results.stats.avg_wait_times[REQUEST_TYPE_ONE]))

            result_load_ratio_1 = results.stats.exp_lambda_1 / results.stats.exp_mu_1
            result_load_ratio_2 = result_load_ratio_1 + results.stats.exp_lambda_2 / results.stats.exp_mu_2

            self.result_frame_label_val.set(f"Расчетная загрузка p1: {results.load_ratio_type_one}\n"
                                            f"Расчетная загрузка p2: {results.load_ratio_type_two}\n"
                                            f"Фактическая загрузка p1': {result_load_ratio_1}\n"
                                            f"Фактическая загрузка p2': {result_load_ratio_2}")
        else:
            # Считывание значений из Spinbox
            lambda1_min = self.lambda1_spinbox_min_val.get()
            lambda1_max = self.lambda1_spinbox_max_val.get()
            lambda2_min = self.lambda2_spinbox_min_val.get()
            lambda2_max = self.lambda2_spinbox_max_val.get()
            mu1_min = self.mu1_spinbox_min_val.get()
            mu1_max = self.mu1_spinbox_max_val.get()
            mu2_min = self.mu2_spinbox_min_val.get()
            mu2_max = self.mu2_spinbox_max_val.get()
            num_requests = self.requests_spinbox_val.get()

            # Открытие нового окна с графиками
            self.plot_window = tk.Toplevel(self.master)
            self.plot_window.title("Результаты моделирования")

            res = run_simulation_passive(
                num_requests,
                {'min': lambda1_min, 'max': lambda1_max, 'step': (lambda1_max - lambda1_min) / 10},
                {'min': lambda2_min, 'max': lambda2_max, 'step': (lambda2_max - lambda2_min) / 10},
                {'min': mu1_min, 'max': mu1_max, 'step': (mu1_max - mu1_min) / 10},
                {'min': mu2_min, 'max': mu2_max, 'step': (mu2_max - mu2_min) / 10}
            )

            # Создание Canvas для графиков
            self.plot_canvas = tk.Canvas(self.plot_window, width=480, height=500)
            self.plot_canvas.grid(row=0, column=0, sticky="nsew")

            # Добавление полосы прокрутки
            self.scrollbar = ttk.Scrollbar(self.plot_window, orient=tk.VERTICAL, command=self.plot_canvas.yview)
            self.scrollbar.grid(row=0, column=1, sticky="ns")

            self.plot_canvas.configure(yscrollcommand=self.scrollbar.set)
            self.plot_canvas.bind('<Configure>', lambda e: self.plot_canvas.configure(scrollregion=self.plot_canvas.bbox("all")))

            # Создание фрейма для размещения графиков
            self.plots_frame = tk.Frame(self.plot_canvas)
            self.plot_canvas.create_window((0, 0), window=self.plots_frame, anchor="nw")
            rc = {"xtick.direction": "inout", "ytick.direction": "inout",
                "xtick.major.size": 5, "ytick.major.size": 5, }
            with plt.rc_context(rc):
                # Создание и добавление графиков
                fig, ax = plt.subplots(figsize=(6, 4))
                # y_values =
                for i in range(5):
                    fig, ax = plt.subplots(figsize=(6, 4))  # Adjust figure size as needed

                    # Prepare data for the line plot
                    y_values = [lambda1_min, lambda1_max, lambda2_min, lambda2_max, mu1_min, mu1_max, mu2_min, mu2_max, num_requests]
                    x_values = np.arange(len(y_values))  # X-coordinates for the values


                    ax.plot(x_values, y_values, marker='o') # Plot a line with circle markers

                    # Setting labels for each point on the line
                    for x, y in zip(x_values, y_values):
                       ax.text(x, y, f"{y:.2f}", ha='center', va='bottom') # Label each point


                    # Setting labels and title
                    ax.set_xlabel("Index")
                    ax.set_ylabel("Value")
                    ax.set_title(f"Line Plot of Variables {i + 1}")


                    # Setting labels and title
                    ax.set_xlabel("Индекс")
                    ax.set_ylabel("Значение")
                    ax.set_title(f"Линейный график переменных {i + 1}")

                    # Настройка стрелок на осях
                    ax.spines['left'].set_position('zero')
                    ax.spines['bottom'].set_position('zero')
                    ax.spines['right'].set_color('none')
                    ax.spines['top'].set_color('none')
                    ax.xaxis.set_ticks_position('bottom')
                    ax.yaxis.set_ticks_position('left')

                    # Adding grid
                    ax.grid(True)

                    # Arrows
                    ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
                            transform=ax.get_yaxis_transform(), clip_on=False)
                    ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                            transform=ax.get_xaxis_transform(), clip_on=False)

                    canvas = FigureCanvasTkAgg(fig, master=self.plots_frame)
                    canvas.draw()
                    canvas.get_tk_widget().grid(row=i, column=0, pady=20)  # Отступы между графиками

            self.plots_frame.update_idletasks()
            self.plot_canvas.config(scrollregion=self.plot_canvas.bbox("all"))

            self.plot_window.rowconfigure(0, weight=1)
            self.plot_window.columnconfigure(0, weight=1)

def main():
    root = tk.Tk()
    ui = UI(root)
    root.mainloop()

if __name__ == "__main__":
    main()