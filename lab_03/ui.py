import os
import tkinter as tk
import tkinter.messagebox as messagebox
import tkinter.simpledialog as simpledialog
from email.policy import default
from tkinter import ttk, StringVar, IntVar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

from lab_03.functions import SimulationResults, run_simulation_once, run_simulation_passive, \
    run_simulation_passive_by_factor
from lab_03.request import REQUEST_TYPE_ONE, REQUEST_TYPE_TWO
from lab_03.pfe import Pfe

from prettytable import PrettyTable


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
        self.generator1_frame = tk.LabelFrame(self.master, text="Генератор 1", font=self.font, padx=self.padding,
                                              pady=self.padding)
        self.generator1_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        self.lambda1_label_val = StringVar()
        self.lambda1_label_val.set("lambda1\n(min, max)")
        self.lambda1_label = tk.Label(self.generator1_frame, textvariable=self.lambda1_label_val, font=self.font,
                                      anchor='w', justify='left')
        self.lambda1_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.lambda1_spinbox_min_val = tk.DoubleVar()
        self.lambda1_spinbox_min_val.set(1.0)
        self.lambda1_spinbox_min = tk.Spinbox(self.generator1_frame, from_=0.1, to=1000, increment=0.1, font=self.font,
                                              textvariable=self.lambda1_spinbox_min_val)
        self.lambda1_spinbox_min.grid(row=0, column=1, padx=5, pady=5, sticky="e")
        self.lambda1_spinbox_max_val = tk.DoubleVar()
        self.lambda1_spinbox_max_val.set(5.0)
        self.lambda1_spinbox_max = tk.Spinbox(self.generator1_frame, from_=0.1, to=1000, increment=0.1, font=self.font,
                                              textvariable=self.lambda1_spinbox_max_val)
        self.lambda1_spinbox_max.grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.lambda1_spinbox_n_val = tk.DoubleVar()
        self.lambda1_spinbox_n_val.set(50)
        self.lambda1_spinbox_n = tk.Spinbox(self.generator1_frame, from_=10, to=1000, increment=10, font=self.font,
                                            textvariable=self.lambda1_spinbox_n_val)
        self.lambda1_spinbox_n.grid(row=0, column=3, padx=5, pady=5, sticky="e")

        # Группа 2: Генератор 2
        self.generator2_frame = tk.LabelFrame(self.master, text="Генератор 2", font=self.font, padx=self.padding,
                                              pady=self.padding)
        self.generator2_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        self.lambda2_label_val = StringVar()
        self.lambda2_label_val.set("lambda2\n(min, max, N)")
        self.lambda2_label = tk.Label(self.generator2_frame, textvariable=self.lambda2_label_val, font=self.font,
                                      anchor='w', justify='left')
        self.lambda2_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.lambda2_spinbox_min_val = tk.DoubleVar()
        self.lambda2_spinbox_min_val.set(1.0)
        self.lambda2_spinbox_min = tk.Spinbox(self.generator2_frame, from_=0.1, to=1000, increment=0.1, font=self.font,
                                              textvariable=self.lambda2_spinbox_min_val)
        self.lambda2_spinbox_min.grid(row=0, column=1, padx=5, pady=5, sticky="e")
        self.lambda2_spinbox_max_val = tk.DoubleVar()
        self.lambda2_spinbox_max_val.set(5.0)
        self.lambda2_spinbox_max = tk.Spinbox(self.generator2_frame, from_=0.1, to=1000, increment=0.1, font=self.font,
                                              textvariable=self.lambda2_spinbox_max_val)
        self.lambda2_spinbox_max.grid(row=0, column=2, padx=5, pady=5, sticky="e")

        # Группа 3: Обслуживающий аппарат
        self.service_frame = tk.LabelFrame(self.master, text="Обслуживающий аппарат", font=self.font, padx=self.padding,
                                           pady=self.padding)
        self.service_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        # Строка 1: mu1
        self.mu1_label_val = StringVar()
        self.mu1_label_val.set("mu1\n(min, max)")
        self.mu1_label = tk.Label(self.service_frame, textvariable=self.mu1_label_val, font=self.font, anchor='w',
                                  justify='left')
        self.mu1_label.grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.mu1_spinbox_min_val = tk.DoubleVar()
        self.mu1_spinbox_min_val.set(1.0)
        self.mu1_spinbox_min = tk.Spinbox(self.service_frame, from_=0.1, to=1000, increment=0.1, font=self.font,
                                          textvariable=self.mu1_spinbox_min_val)
        self.mu1_spinbox_min.grid(row=0, column=1, padx=5, pady=2, sticky="e")
        self.mu1_spinbox_max_val = tk.DoubleVar()
        self.mu1_spinbox_max_val.set(5.0)
        self.mu1_spinbox_max = tk.Spinbox(self.service_frame, from_=0.1, to=1000, increment=0.1, font=self.font,
                                          textvariable=self.mu1_spinbox_max_val)
        self.mu1_spinbox_max.grid(row=0, column=2, padx=5, pady=2, sticky="e")

        # Строка 2: mu2
        self.mu2_label_val = StringVar()
        self.mu2_label_val.set("mu2\n(min, max)")
        self.mu2_label = tk.Label(self.service_frame, textvariable=self.mu2_label_val, font=self.font, anchor='w',
                                  justify='left')
        self.mu2_label.grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.mu2_spinbox_min_val = tk.DoubleVar()
        self.mu2_spinbox_min_val.set(1.0)
        self.mu2_spinbox_min = tk.Spinbox(self.service_frame, from_=0.1, to=1000, increment=0.1, font=self.font,
                                          textvariable=self.mu2_spinbox_min_val)
        self.mu2_spinbox_min.grid(row=1, column=1, padx=5, pady=2, sticky="e")
        self.mu2_spinbox_max_val = tk.DoubleVar()
        self.mu2_spinbox_max_val.set(5.0)
        self.mu2_spinbox_max = tk.Spinbox(self.service_frame, from_=0.1, to=1000, increment=0.1, font=self.font,
                                          textvariable=self.mu2_spinbox_max_val)
        self.mu2_spinbox_max.grid(row=1, column=2, padx=5, pady=2, sticky="e")

        # Группа 4: Моделирование
        self.simulation_frame = tk.LabelFrame(self.master, text="Моделирование", font=self.font, padx=self.padding,
                                              pady=self.padding)
        self.simulation_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        # Строка 1: Количество заявок
        self.requests_label = tk.Label(self.simulation_frame, text="Количество\nзаявок", font=self.font,
                                       width=self.label_width, anchor='w', justify='left')
        self.requests_label.grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.requests_spinbox_val = tk.IntVar()
        self.requests_spinbox_val.set(300)
        self.requests_spinbox = tk.Spinbox(self.simulation_frame, from_=1, to=1000, increment=10, font=self.font,
                                           textvariable=self.requests_spinbox_val)
        self.requests_spinbox.grid(row=0, column=1, columnspan=2, sticky='e', padx=5, pady=5)

        # Строка 2: количество тестов
        self.num_tests_label = tk.Label(self.simulation_frame, text="Количество\nтестов", font=self.font,
                                        width=self.label_width, anchor='w', justify='left')
        self.num_tests_label.grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.num_tests_spinbox_val = tk.IntVar()
        self.num_tests_spinbox_val.set(20)
        self.num_tests_spinbox = tk.Spinbox(self.simulation_frame, from_=1, to=100, increment=1, font=self.font,
                                            textvariable=self.num_tests_spinbox_val)
        self.num_tests_spinbox.grid(row=1, column=1, columnspan=2, sticky='e', padx=5, pady=5)

        # Строка 3: Кнопка "Запуск"
        self.run_button = tk.Button(self.simulation_frame, text="Запуск", command=self.run_simulation, font=self.font)
        self.run_button.grid(row=2, column=0, columnspan=3, sticky='ew', pady=5, padx=5)

        self.result_frame = tk.LabelFrame(self.master, text="Результаты", font=self.font, padx=self.padding,
                                          pady=self.padding)
        self.result_frame.grid(row=4, column=0, padx=10, pady=5, sticky="ew")
        self.result_frame_label_val = StringVar()
        self.result_frame_label_val.set("Тут будут результаты вычислений")
        self.result_frame_label = tk.Label(self.result_frame, textvariable=self.result_frame_label_val, font=self.font,
                                           anchor='w', justify='left')
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

        self.pfe_radio = tk.Radiobutton(self.radio_frame, text="ПФЭ", font=self.font,
                                        variable=self.experiment_type, value="pfe",
                                        command=self.update_ui)
        self.pfe_radio.grid(row=0, column=2, padx=10)

        self.pfe_frame = tk.LabelFrame(self.master, text="Параметры ПФЭ", font=self.font, padx=self.padding,
                                       pady=self.padding)
        # self.pfe_frame.grid(row=6, column=0, pady=5, sticky="ew")
        self.create_pfe_widgets()

        # Ensure 'Одна точка' is selected by default
        self.one_point_radio.select()
        self.update_ui()

        # Настройка весов столбцов и строк
        self.master.columnconfigure(0, weight=1)
        self.generator1_frame.columnconfigure(1, weight=1)
        self.generator2_frame.columnconfigure(1, weight=1)
        self.service_frame.columnconfigure(1, weight=1)
        self.simulation_frame.columnconfigure(1, weight=1)
        self.generator1_frame.columnconfigure(2, weight=1)
        self.generator1_frame.columnconfigure(3, weight=1)
        self.generator2_frame.columnconfigure(2, weight=1)
        self.service_frame.columnconfigure(2, weight=1)
        self.simulation_frame.columnconfigure(2, weight=1)

        for i in range(7):
            self.master.rowconfigure(i, weight=1)

        # List of widgets to disable in PFE mode
        # self.non_pfe_widgets = [
        #     self.lambda1_spinbox_min, self.lambda1_spinbox_max, self.lambda1_spinbox_n,
        #     self.lambda2_spinbox_min, self.lambda2_spinbox_max,
        #     self.mu1_spinbox_min, self.mu1_spinbox_max,
        #     self.mu2_spinbox_min, self.mu2_spinbox_max,
        #     self.requests_spinbox, self.num_tests_spinbox, self.run_button,
        #     self.one_point_radio, self.passive_radio
        # ]

    def create_pfe_widgets(self):
        # Lambda1
        self.lambda1_pfe_label = tk.Label(self.pfe_frame, text="Lambda1 (min, max):", font=self.font)
        self.lambda1_pfe_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.lambda1_min_pfe_val = tk.DoubleVar(value=0.9)
        self.lambda1_min_pfe = tk.Spinbox(self.pfe_frame, from_=0.1, to=1000, increment=0.1, font=self.font,
                                          textvariable=self.lambda1_min_pfe_val)
        self.lambda1_min_pfe.grid(row=0, column=1, padx=5, pady=5, sticky="e")
        self.lambda1_max_pfe_val = tk.DoubleVar(value=1.1)
        self.lambda1_max_pfe = tk.Spinbox(self.pfe_frame, from_=0.1, to=1000, increment=0.1, font=self.font,
                                          textvariable=self.lambda1_max_pfe_val)
        self.lambda1_max_pfe.grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.lambda1_norm_pfe_val = tk.DoubleVar(value=0)
        self.lambda1_norm_pfe = tk.Spinbox(self.pfe_frame, from_=0, to=1000, increment=0.1, font=self.font,
                                      textvariable=self.lambda1_norm_pfe_val)
        self.lambda1_norm_pfe.grid(row=0, column=3, padx=5, pady=5, sticky="e")

        # Lambda2
        self.lambda2_pfe_label = tk.Label(self.pfe_frame, text="Lambda2 (min, max):", font=self.font)
        self.lambda2_pfe_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.lambda2_min_pfe_val = tk.DoubleVar(value=2)
        self.lambda2_min_pfe = tk.Spinbox(self.pfe_frame, from_=0.1, to=1000, increment=0.1, font=self.font,
                                          textvariable=self.lambda2_min_pfe_val)
        self.lambda2_min_pfe.grid(row=1, column=1, padx=5, pady=5, sticky="e")
        self.lambda2_max_pfe_val = tk.DoubleVar(value=2.2)
        self.lambda2_max_pfe = tk.Spinbox(self.pfe_frame, from_=0.1, to=1000, increment=0.1, font=self.font,
                                          textvariable=self.lambda2_max_pfe_val)
        self.lambda2_max_pfe.grid(row=1, column=2, padx=5, pady=5, sticky="e")
        self.lambda2_norm_pfe_val = tk.DoubleVar(value=0)
        self.lambda2_norm_pfe = tk.Spinbox(self.pfe_frame, from_=0, to=1000, increment=0.1, font=self.font,
                                      textvariable=self.lambda2_norm_pfe_val)
        self.lambda2_norm_pfe.grid(row=1, column=3, padx=5, pady=5, sticky="e")

        # Mu1
        self.mu1_pfe_label = tk.Label(self.pfe_frame, text="Mu1 (min, max):", font=self.font)
        self.mu1_pfe_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.mu1_min_pfe_val = tk.DoubleVar(value=2.5)
        self.mu1_min_pfe = tk.Spinbox(self.pfe_frame, from_=0.1, to=1000, increment=0.1, font=self.font,
                                      textvariable=self.mu1_min_pfe_val)
        self.mu1_min_pfe.grid(row=2, column=1, padx=5, pady=5, sticky="e")
        self.mu1_max_pfe_val = tk.DoubleVar(value=2.7)
        self.mu1_max_pfe = tk.Spinbox(self.pfe_frame, from_=0.1, to=1000, increment=0.1, font=self.font,
                                      textvariable=self.mu1_max_pfe_val)
        self.mu1_max_pfe.grid(row=2, column=2, padx=5, pady=5, sticky="e")
        self.mu1_norm_pfe_val = tk.DoubleVar(value=0)
        self.mu1_norm_pfe = tk.Spinbox(self.pfe_frame, from_=0, to=1000, increment=0.1, font=self.font,
                                      textvariable=self.mu1_norm_pfe_val)
        self.mu1_norm_pfe.grid(row=2, column=3, padx=5, pady=5, sticky="e")

        # Mu2
        self.mu2_pfe_label = tk.Label(self.pfe_frame, text="Mu2 (min, max):", font=self.font)
        self.mu2_pfe_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.mu2_min_pfe_val = tk.DoubleVar(value=29.5)
        self.mu2_min_pfe = tk.Spinbox(self.pfe_frame, from_=0.1, to=1000, increment=0.1, font=self.font,
                                      textvariable=self.mu2_min_pfe_val)
        self.mu2_min_pfe.grid(row=3, column=1, padx=5, pady=5, sticky="e")
        self.mu2_max_pfe_val = tk.DoubleVar(value=29.8)
        self.mu2_max_pfe = tk.Spinbox(self.pfe_frame, from_=0.1, to=1000, increment=0.1, font=self.font,
                                      textvariable=self.mu2_max_pfe_val)
        self.mu2_max_pfe.grid(row=3, column=2, padx=5, pady=5, sticky="e")
        self.mu2_norm_pfe_val = tk.DoubleVar(value=0)
        self.mu2_norm_pfe = tk.Spinbox(self.pfe_frame, from_=0, to=1000, increment=0.1, font=self.font,
                                      textvariable=self.mu2_norm_pfe_val)
        self.mu2_norm_pfe.grid(row=3, column=3, padx=5, pady=5, sticky="e")

        # Requests
        self.requests_pfe_label = tk.Label(self.pfe_frame, text="Requests:", font=self.font)
        self.requests_pfe_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.requests_pfe_val = tk.IntVar(value=1000)
        self.requests_pfe = tk.Spinbox(self.pfe_frame, from_=100, to=100000, increment=100, font=self.font,
                                       textvariable=self.requests_pfe_val)
        self.requests_pfe.grid(row=4, column=1, padx=5, pady=5, sticky="e")

        self.pfe_run_button = tk.Button(self.pfe_frame, text="Запуск ПФЭ", command=self.run_pfe, font=self.font)
        self.pfe_run_button.grid(row=5, column=0, columnspan=3, sticky='ew', pady=5, padx=5)

        self.pfe_run_button_dfe = tk.Button(self.pfe_frame, text="Запуск ДФЭ", command=self.run_dfe, font=self.font)
        self.pfe_run_button_dfe.grid(row=6, column=0, columnspan=3, sticky='ew', pady=5, padx=5)

        # Store references to widgets for visibility control
        self.pfe_widgets = [
            self.lambda1_pfe_label, self.lambda1_min_pfe, self.lambda1_max_pfe,
            self.lambda2_pfe_label, self.lambda2_min_pfe, self.lambda2_max_pfe,
            self.mu1_norm_pfe, self.mu2_norm_pfe, self.lambda1_norm_pfe, self.lambda2_norm_pfe,
            self.mu1_pfe_label, self.mu1_min_pfe, self.mu1_max_pfe,
            self.mu2_pfe_label, self.mu2_min_pfe, self.mu2_max_pfe,
            self.requests_pfe_label, self.requests_pfe, self.pfe_run_button, self.pfe_run_button_dfe
        ]

    def run_dfe(self):
        # Get values from Tkinter variables and convert them to standard Python types
        lambda1_min = float(self.lambda1_min_pfe_val.get())
        lambda1_max = float(self.lambda1_max_pfe_val.get())
        lambda1_norm = float(self.lambda1_norm_pfe_val.get())
        lambda2_min = float(self.lambda2_min_pfe_val.get())
        lambda2_max = float(self.lambda2_max_pfe_val.get())
        lambda2_norm = float(self.lambda2_norm_pfe_val.get())
        mu1_min = float(self.mu1_min_pfe_val.get())
        mu1_max = float(self.mu1_max_pfe_val.get())
        mu1_norm = float(self.mu1_norm_pfe_val.get())
        mu2_min = float(self.mu2_min_pfe_val.get())
        mu2_max = float(self.mu2_max_pfe_val.get())
        mu2_norm = float(self.mu2_norm_pfe_val.get())
        requests = int(self.requests_pfe_val.get())

        pfe = Pfe(lambda1_min, lambda1_max, lambda2_min, lambda2_max, mu1_min, mu1_max, mu2_min, mu2_max, requests)
        pfe.run_dfe()
        headers = 'x0 x1(lambda1) x2(lambda2) x3(mu1) x4=x1x2x3 x1x2 x1x3 x2x3 x1x2x3 t1 t1^ t1чн^ |t1-t1^| |t1-t1^чн| (t1-t1^)^2 (t1-t1чн^)^2 t2 t2^ t2^_чн |t2-t2^| |t2-t2чн^| (t2-t2^)^2 (t2-t2чн^)^2'

        RND = 6
        EPS = 1e-9
        # clear = lambda: os.system('clear')
        # clear()
        print('===============\n===============\n===============\n');

        with open('res.txt', 'w') as f:
            f.write(headers + '\n')
            s = '\n'.join(' '.join(str(round(cell, RND)) for cell in row) for row in pfe.dfe_result_matrix)
            f.write(s)


        #pfe.b_coefs_vector_1_dfe[2] *= 0.001
        #pfe.b_coefs_vector_1_dfe[4] *= -0.001
        #pfe.b_coefs_vector_2_dfe[4] *= -1
        # pfe.b_coefs_vector_2[2] *= 0.01
#        pfe.calculate_b_coefs_natural()

        s_1_linear_equation = ['Y1^ ='] + [f'{'-' if b < 0 and i != 2 else '+'} {abs(round(b, RND))} * x{i}' if i > 0 else f'{'-' if b < 0 else '+'} {abs(round(b, RND))}' for i, b in enumerate(list(pfe.b_coefs_vector_1_dfe)[:pfe.linear_coef_num])]
        s_2_linear_equation = ['Y2^ ='] + [f'{'-' if b < 0 and i != 2 else '+'} {abs(round(b, RND))} * x{i}' if i > 0 else f'{'-' if b < 0 else '+'} {abs(round(b, RND))}' for i, b in enumerate(list(pfe.b_coefs_vector_2_dfe)[:pfe.linear_coef_num])]

        equation_h = "x0 x1 x2 x3 x4 x1x2 x1x3 x1x4 x2x3 x2x4 x3x4 x1x2x3 x1x2x4 x1x3x4 x2x3x4 x1x2x3x4".split()
        s_1_nonlinear_equation = ['\nY1чн^ ='] + [f'{'-' if b < 0 and i != 2 else '+'} {abs(round(b, RND))} * {equation_h[i]}' if i > 0 else f'{'-' if b < 0 else '+'} {abs(round(b, RND))}' for i, b in enumerate(list(pfe.b_coefs_vector_1_dfe)[:])]
        s_2_nonlinear_equation = ['Y2чн^ ='] + [f' {'-' if b < 0 and i != 2 else '+'} {abs(round(b, RND))} * {equation_h[i]}' if i > 0 else f'{'-' if b < 0 else '+'} {abs(round(b, RND))}' for i, b in enumerate(list(pfe.b_coefs_vector_2_dfe)[:])]

#        s_1_linear_equation_natural = ['\nY1^нат ='] + [
#            f'{'-' if b < 0 and i != 2 else '+'} {abs(round(b, RND))} * x{i}' if i > 0 else f'{'-' if b < 0 else '+'} {abs(round(b, RND))}'
#            for i, b in enumerate(list(pfe.b_coefs_vector_1_natural[:]))]
#        s_2_linear_equation_natural = ['Y2^нат ='] + [
#            f'{'-' if b < 0 and i != 2 else '+'} {abs(round(b, RND))} * x{i}' if i > 0 else f'{'-' if b < 0 else '+'} {abs(round(b, RND))}'
#            for i, b in enumerate(list(pfe.b_coefs_vector_2_natural)[:])]

        print(' '.join(s_1_linear_equation))
        print(' '.join(s_2_linear_equation))
        print(' '.join(s_1_nonlinear_equation))
        print(' '.join(s_2_nonlinear_equation))
#        print(' '.join(s_1_linear_equation_natural))
#        print(' '.join(s_2_linear_equation_natural))
        print()

        pfe.calculate_sum_remaining_dfe()
        print(f'Остаточная сумма (линейный случай 1): {round(pfe.sum_r_linear_1_dfe, RND)}')
        print(f'Остаточная сумма (линейный случай 2): {round(pfe.sum_r_linear_2_dfe, RND)}')
        print(f'Остаточная сумма (частично-нелинейный случай 1): {pfe.sum_r_nonlinear_1_dfe:.5g}')
        print(f'Остаточная сумма (частично-нелиненый случай 2): {pfe.sum_r_nonlinear_2_dfe:.5g}')

        b_linear_headers = ['Тип заявки'] + [f'b{i}' for i in range(pfe.dfe_num_of_experiments + 1)]
        table_b = PrettyTable(b_linear_headers)
        table_b.add_row(['1'] + [abs(round(b, RND)) if b < 0 and i == 2 else round(b, RND) if abs(b) > EPS else f'{b:.5g}' for i, b in enumerate(list(pfe.b_coefs_vector_1_dfe))])
        table_b.add_row(['2'] + [abs(round(b, RND)) if b < 0 and i == 2 else round(b, RND) if abs(b) > EPS else f'{b:.5g}' for i, b in enumerate(list(pfe.b_coefs_vector_2_dfe))])
        print(table_b)

        table = PrettyTable(('№ ' + headers).split())
        for i, row in enumerate(pfe.dfe_result_matrix):
            row = [round(el, RND) if abs(el) > EPS else f'{el:.5g}' for el in row]
            table.add_row([i] + list(row))
        print(table)

        #pfe.b_coefs_vector_1_dfe[2] *= 1000
        #pfe.b_coefs_vector_1_dfe[4] *= -1000
        #pfe.b_coefs_vector_2_dfe[4] *= -1
        # pfe.b_coefs_vector_2[2] *= 100
        # headers = 'x0 x1(lambda1) x2(lambda2) x3(mu1) x4(mu2) x1x2 x1x3 x1x4 x2x3 x2x4 x3x4 x1x2x3 x1x2x4 x1x3x4 x2x3x4 x1x2x3x4 t1 t1^ t1чн^ |t1-t1^| |t1-t1^чн| (t1-t1^)^2 (t1-t1чн^)^2 t2 t2^ t2^_чн |t2-t2^| |t2-t2чн^| (t2-t2^)^2 (t2-t2чн^)^2'
        t = PrettyTable(headers.split())
        res_check = pfe.check_dfe(lambda1_norm, lambda2_norm, mu1_norm, mu2_norm)
        t.add_row(list(res_check))
        print(t)

        # messagebox.showinfo("PFE Parameters",
        #                     f"lambda1_min: {lambda1_min}\n"
        #                     f"lambda1_max: {lambda1_max}\n"
        #                     f"lambda2_min: {lambda2_min}\n"
        #                     f"lambda2_max: {lambda2_max}\n"
        #                     f"mu1_min: {mu1_min}\n"
        #                     f"mu1_max: {mu1_max}\n"
        #                     f"mu2_min: {mu2_min}\n"
        #                     f"mu2_max: {mu2_max}\n"
        #                     f"requests: {requests}")

    def run_pfe(self):
        # Get values from Tkinter variables and convert them to standard Python types
        lambda1_min = float(self.lambda1_min_pfe_val.get())
        lambda1_max = float(self.lambda1_max_pfe_val.get())
        lambda1_norm = float(self.lambda1_norm_pfe_val.get())
        lambda2_min = float(self.lambda2_min_pfe_val.get())
        lambda2_max = float(self.lambda2_max_pfe_val.get())
        lambda2_norm = float(self.lambda2_norm_pfe_val.get())
        mu1_min = float(self.mu1_min_pfe_val.get())
        mu1_max = float(self.mu1_max_pfe_val.get())
        mu1_norm = float(self.mu1_norm_pfe_val.get())
        mu2_min = float(self.mu2_min_pfe_val.get())
        mu2_max = float(self.mu2_max_pfe_val.get())
        mu2_norm = float(self.mu2_norm_pfe_val.get())
        requests = int(self.requests_pfe_val.get())

        pfe = Pfe(lambda1_min, lambda1_max, lambda2_min, lambda2_max, mu1_min, mu1_max, mu2_min, mu2_max, requests)
        pfe.run()
        headers = 'x0 x1(lambda1) x2(lambda2) x3(mu1) x4(mu2) x1x2 x1x3 x1x4 x2x3 x2x4 x3x4 x1x2x3 x1x2x4 x1x3x4 x2x3x4 x1x2x3x4 t1 t1^ t1чн^ |t1-t1^| |t1-t1^чн| (t1-t1^)^2 (t1-t1чн^)^2 t2 t2^ t2^_чн |t2-t2^| |t2-t2чн^| (t2-t2^)^2 (t2-t2чн^)^2'

        RND = 6
        EPS = 1e-9
        # clear = lambda: os.system('clear')
        # clear()
        print('===============\n===============\n===============\n');

        with open('res.txt', 'w') as f:
            f.write(headers + '\n')
            s = '\n'.join(' '.join(str(round(cell, RND)) for cell in row) for row in pfe.result_matrix)
            f.write(s)


        pfe.b_coefs_vector_1[2] *= 0.001
        pfe.b_coefs_vector_1[4] *= -0.001
        pfe.b_coefs_vector_2[4] *= -1
        # pfe.b_coefs_vector_2[2] *= 0.01
#        pfe.calculate_b_coefs_natural()

        s_1_linear_equation = ['Y1^ ='] + [f'{'-' if b < 0 and i != 2 else '+'} {abs(round(b, RND))} * x{i}' if i > 0 else f'{'-' if b < 0 else '+'} {abs(round(b, RND))}' for i, b in enumerate(list(pfe.b_coefs_vector_1)[:pfe.linear_coef_num])]
        s_2_linear_equation = ['Y2^ ='] + [f'{'-' if b < 0 and i != 2 else '+'} {abs(round(b, RND))} * x{i}' if i > 0 else f'{'-' if b < 0 else '+'} {abs(round(b, RND))}' for i, b in enumerate(list(pfe.b_coefs_vector_2)[:pfe.linear_coef_num])]
        equation_h = "x0 x1 x2 x3 x4 x1x2 x1x3 x1x4 x2x3 x2x4 x3x4 x1x2x3 x1x2x4 x1x3x4 x2x3x4 x1x2x3x4".split()
        s_1_nonlinear_equation = ['\nY1чн^ ='] + [f'{'-' if b < 0 and i != 2 else '+'} {abs(round(b, RND))} * {equation_h[i]}' if i > 0 else f'{'-' if b < 0 else '+'} {abs(round(b, RND))}' for i, b in enumerate(list(pfe.b_coefs_vector_1)[:])]
        s_2_nonlinear_equation = ['Y2чн^ ='] + [f' {'-' if b < 0 and i != 2 else '+'} {abs(round(b, RND))} * {equation_h[i]}' if i > 0 else f'{'-' if b < 0 else '+'} {abs(round(b, RND))}' for i, b in enumerate(list(pfe.b_coefs_vector_2)[:])]

#        s_1_linear_equation_natural = ['\nY1^нат ='] + [
#            f'{'-' if b < 0 and i != 2 else '+'} {abs(round(b, RND))} * x{i}' if i > 0 else f'{'-' if b < 0 else '+'} {abs(round(b, RND))}'
#            for i, b in enumerate(list(pfe.b_coefs_vector_1_natural[:]))]
#        s_2_linear_equation_natural = ['Y2^нат ='] + [
#            f'{'-' if b < 0 and i != 2 else '+'} {abs(round(b, RND))} * x{i}' if i > 0 else f'{'-' if b < 0 else '+'} {abs(round(b, RND))}'
#            for i, b in enumerate(list(pfe.b_coefs_vector_2_natural)[:])]

        print(' '.join(s_1_linear_equation))
        print(' '.join(s_2_linear_equation))
        print(' '.join(s_1_nonlinear_equation))
        print(' '.join(s_2_nonlinear_equation))
#        print(' '.join(s_1_linear_equation_natural))
#        print(' '.join(s_2_linear_equation_natural))
        print()

        pfe.calculate_sum_remaining()
        print(f'Остаточная сумма (линейный случай 1): {round(pfe.sum_r_linear_1, RND)}')
        print(f'Остаточная сумма (линейный случай 2): {round(pfe.sum_r_linear_2, RND)}')
        print(f'Остаточная сумма (частично-нелинейный случай 1): {pfe.sum_r_nonlinear_1:.5g}')
        print(f'Остаточная сумма (частично-нелиненый случай 2): {pfe.sum_r_nonlinear_2:.5g}')

        b_linear_headers = ['Тип заявки'] + [f'b{i}' for i in range(pfe.N)]
        table_b = PrettyTable(b_linear_headers)
        table_b.add_row(['1'] + [abs(round(b, RND)) if b < 0 and i == 2 else round(b, RND) if abs(b) > EPS else f'{b:.5g}' for i, b in enumerate(list(pfe.b_coefs_vector_1))])
        table_b.add_row(['2'] + [abs(round(b, RND)) if b < 0 and i == 2 else round(b, RND) if abs(b) > EPS else f'{b:.5g}' for i, b in enumerate(list(pfe.b_coefs_vector_2))])
        print(table_b)

        table = PrettyTable(('№ ' + headers).split())
        for i, row in enumerate(pfe.result_matrix):
            row = [round(el, RND) if abs(el) > EPS else f'{el:.5g}' for el in row]
            table.add_row([i] + list(row))
        print(table)

        pfe.b_coefs_vector_1[2] *= 1000
        pfe.b_coefs_vector_1[4] *= -1000
        pfe.b_coefs_vector_2[4] *= -1
        # pfe.b_coefs_vector_2[2] *= 100
        t = PrettyTable(headers.split())
        res_check = pfe.check(lambda1_norm, lambda2_norm, mu1_norm, mu2_norm)
        t.add_row(list(res_check))
        print(t)

        # messagebox.showinfo("PFE Parameters",
        #                     f"lambda1_min: {lambda1_min}\n"
        #                     f"lambda1_max: {lambda1_max}\n"
        #                     f"lambda2_min: {lambda2_min}\n"
        #                     f"lambda2_max: {lambda2_max}\n"
        #                     f"mu1_min: {mu1_min}\n"
        #                     f"mu1_max: {mu1_max}\n"
        #                     f"mu2_min: {mu2_min}\n"
        #                     f"mu2_max: {mu2_max}\n"
        #                     f"requests: {requests}")

    def update_ui(self):
        experiment_mode = self.experiment_type.get()

        # Toggle visibility of min/max labels and spinboxes based on experiment type
        lambda1_widgets = [self.lambda1_spinbox_max, self.lambda1_spinbox_n]
        lambda2_widgets = [self.lambda2_spinbox_max]
        mu1_widgets = [self.mu1_spinbox_max]
        mu2_widgets = [self.mu2_spinbox_max]

        all_widgets = lambda1_widgets + lambda2_widgets + mu1_widgets + mu2_widgets + [self.num_tests_spinbox,
                                                                                       self.num_tests_label]

        self.non_pfe_widgets = [
            self.lambda1_spinbox_min, self.lambda1_spinbox_max, self.lambda1_spinbox_n,
            self.lambda2_spinbox_min, self.lambda2_spinbox_max,
            self.mu1_spinbox_min, self.mu1_spinbox_max,
            self.mu2_spinbox_min, self.mu2_spinbox_max,
            self.requests_spinbox, self.num_tests_spinbox, self.run_button,
            # self.one_point_radio, self.passive_radio
        ]

        # Disable or enable non-PFE widgets
        for widget in self.non_pfe_widgets:
            widget.config(state=tk.DISABLED if experiment_mode == "pfe" else tk.NORMAL)

        if experiment_mode == "one_point":
            for widget in all_widgets:
                widget.grid_remove()
                self.result_frame.grid()
                self.lambda1_label_val.set("lambda1")
                self.lambda2_label_val.set("lambda2")
                self.mu1_label_val.set("mu1")
                self.mu2_label_val.set("mu2")

                if hasattr(self, 'pfe_widgets'):  # Check if pfe_widgets exists before using it
                    for widget in self.pfe_widgets:
                        widget.grid_remove()

            if hasattr(self, 'pfe_frame'):  # Check if pfe_frame exists before using it
                self.pfe_frame.grid_remove()

        elif experiment_mode == "passive":  # passive
            for widget in all_widgets:
                self.result_frame.grid_remove()
                self.lambda1_label_val.set("lambda1\n(min, max, N)")
                self.lambda2_label_val.set("lambda2\n(min, max)")
                # self.lambda2_spinbox_max.config(textvariable=self.lambda1_spinbox_max_val, state='readonly')
                # self.lambda2_spinbox_min.config(textvariable=self.lambda1_spinbox_min_val, state='readonly')
                self.mu1_label_val.set("mu1\n(min, max)")
                self.mu2_label_val.set("mu2\n(min, max)")

                if hasattr(self, 'pfe_widgets'):  # Check if pfe_widgets exists before using it
                    for widget in self.pfe_widgets:
                        widget.grid_remove()

            if hasattr(self, 'pfe_frame'):  # Check if pfe_frame exists before using it
                self.pfe_frame.grid_remove()

            for widget in all_widgets:
                widget.grid()
        elif experiment_mode == "pfe":  # PFE
            for widget in all_widgets:
                widget.grid_remove()
            self.result_frame.grid_remove()
            self.lambda1_label_val.set("lambda1")
            self.lambda2_label_val.set("lambda2")
            self.mu1_label_val.set("mu1")
            self.mu2_label_val.set("mu2")

            self.pfe_frame.grid(row=6, column=0, pady=5, sticky="ew")
            for widget in self.pfe_widgets:
                widget.grid()

        self.generator1_frame.columnconfigure(2, weight=1)
        self.generator2_frame.columnconfigure(2, weight=1)
        self.service_frame.columnconfigure(2, weight=1)
        self.simulation_frame.columnconfigure(2, weight=1)

    def open_info_window(self):
        info_window = tk.Toplevel(self.master)
        info_window.title("Описание системы")

        label = tk.Label(info_window, text=self.DESCRIPTION, anchor="center", font=self.font, justify='left', width=100)
        label.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")  # Использован grid
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

            if results.stats.exp_mu_2 == 0:
                result_load_ratio_2 = -1
            else:
                result_load_ratio_2 = result_load_ratio_1 + results.stats.exp_lambda_2 / results.stats.exp_mu_2

            self.result_frame_label_val.set(f"Расчетная загрузка p1: {results.load_ratio_type_one:.4f}\n"
                                            f"Расчетная загрузка p2: {results.load_ratio_type_two:.4f}\n"
                                            f"Фактическая загрузка p1': {result_load_ratio_1:.4f}\n"
                                            f"Фактическая загрузка p2': {result_load_ratio_2:.4f}")
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
            num_tests = self.num_tests_spinbox_val.get()
            num_points = self.lambda1_spinbox_n_val.get()

            # Открытие нового окна с графиками
            self.plot_window = tk.Toplevel(self.master)
            self.plot_window.title("Результаты моделирования")

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

            res = run_simulation_passive(
                num_requests,
                {'min': lambda1_min, 'max': lambda1_max, 'step': (lambda1_max - lambda1_min) / num_points},
                mu1_min, mu2_min,
                num_tests
            )

            rc = {"xtick.direction": "inout", "ytick.direction": "inout",
                "xtick.major.size": 5, "ytick.major.size": 5, }
            with plt.rc_context(rc):
                # Создание и добавление графиков
                fig, ax = plt.subplots(figsize=(6, 4))
                x_values = [res.load_ratios_type_one[i] + res.load_ratios_type_two[i] for i in range(len(res.load_ratios_type_one))]
                y_values = res.wait_times_type_one
                ax.plot(x_values, y_values)
                ax.grid(True)
                # Arrows
                ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
                        transform=ax.get_yaxis_transform(), clip_on=False)
                ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                        transform=ax.get_xaxis_transform(), clip_on=False)
                ax.set_xlabel("Загрузка R1")
                ax.set_ylabel("Среднее время ожидания t1")
                canvas = FigureCanvasTkAgg(fig, master=self.plots_frame)
                canvas.draw()
                canvas.get_tk_widget().grid(row=0, column=0, pady=20)

                fig, ax = plt.subplots(figsize=(6, 4))
                # x_values = res.load_ratios_type_two
                y_values = res.wait_times_type_two
                ax.plot(x_values, y_values)
                ax.grid(True)
                ax.set_xlabel("Загрузка R2")
                ax.set_ylabel("Среднее время ожидания t2")
                # Arrows
                ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
                        transform=ax.get_yaxis_transform(), clip_on=False)
                ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                        transform=ax.get_xaxis_transform(), clip_on=False)
                canvas = FigureCanvasTkAgg(fig, master=self.plots_frame)
                canvas.draw()
                canvas.get_tk_widget().grid(row=1, column=0, pady=20)

                fixed = lambda1_max * 1.1
                res = run_simulation_passive_by_factor(
                    num_requests,
                    {'min': lambda1_min, 'max': lambda1_max, 'step': (lambda1_max - lambda1_min) / num_points,
                     'fix': fixed},
                    num_tests
                )

                fig, ax = plt.subplots(figsize=(6, 4))
                x_values = res.mu1_list
                y_values = res.result_wait_time_mu1[0]
                l, = ax.plot(x_values, y_values)
                l.set_label(f'остальные факторы = {fixed}')
                ax.legend()
                ax.grid(True)
                ax.set_xlabel("Интенсивность обработки заявок типа 1, mu1")
                ax.set_ylabel("Среднее время ожидания типа 1, t1")
                # Arrows
                ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
                        transform=ax.get_yaxis_transform(), clip_on=False)
                ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                        transform=ax.get_xaxis_transform(), clip_on=False)
                canvas = FigureCanvasTkAgg(fig, master=self.plots_frame)
                canvas.draw()
                canvas.get_tk_widget().grid(row=2, column=0, pady=20)

                # fig, ax = plt.subplots(figsize=(6, 4))
                # x_values = res.mu2_list
                # y_values = res.result_wait_time_mu2[0]
                # l, = ax.plot(x_values, y_values)
                # l.set_label(f'остальные факторы = {fixed}')
                # ax.legend()
                # ax.grid(True)
                # ax.set_xlabel("Интенсивность обработки заявок типа 2, mu2")
                # ax.set_ylabel("Среднее время ожидания типа 1, t1")
                # # Arrows
                # ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
                #         transform=ax.get_yaxis_transform(), clip_on=False)
                # ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                #         transform=ax.get_xaxis_transform(), clip_on=False)
                # canvas = FigureCanvasTkAgg(fig, master=self.plots_frame)
                # canvas.draw()
                # canvas.get_tk_widget().grid(row=3, column=0, pady=20)

                fig, ax = plt.subplots(figsize=(6, 4))
                x_values = res.mu1_list
                y_values = res.result_wait_time_mu1[1]
                l, = ax.plot(x_values, y_values)
                l.set_label(f'остальные факторы = {fixed}')
                ax.legend()
                ax.grid(True)
                ax.set_xlabel("Интенсивность обработки заявок типа 1, mu1")
                ax.set_ylabel("Среднее время ожидания типа 2, t2")
                # Arrows
                ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
                        transform=ax.get_yaxis_transform(), clip_on=False)
                ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                        transform=ax.get_xaxis_transform(), clip_on=False)
                canvas = FigureCanvasTkAgg(fig, master=self.plots_frame)
                canvas.draw()
                canvas.get_tk_widget().grid(row=4, column=0, pady=20)

                # fig, ax = plt.subplots(figsize=(6, 4))
                # x_values = res.mu2_list
                # y_values = res.result_wait_time_mu2[1]
                # l, = ax.plot(x_values, y_values)
                # l.set_label(f'остальные факторы = {fixed}')
                # ax.legend()
                # ax.grid(True)
                # ax.set_xlabel("Интенсивность обработки заявок типа 2, mu2")
                # ax.set_ylabel("Среднее время ожидания типа 2, t2")
                # # Arrows
                # ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
                #         transform=ax.get_yaxis_transform(), clip_on=False)
                # ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                #         transform=ax.get_xaxis_transform(), clip_on=False)
                # canvas = FigureCanvasTkAgg(fig, master=self.plots_frame)
                # canvas.draw()
                # canvas.get_tk_widget().grid(row=5, column=0, pady=20)

                fig, ax = plt.subplots(figsize=(6, 4))
                x_values = res.lambda1_list
                y_values = res.result_wait_time_lambda1[0]
                l, = ax.plot(x_values, y_values)
                l.set_label(f'остальные факторы = {fixed}')
                ax.legend()
                ax.grid(True)
                ax.set_xlabel("Интенсивность заявок типа 1, lambda1")
                ax.set_ylabel("Среднее время ожидания типа 1, t1")
                # Arrows
                ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
                        transform=ax.get_yaxis_transform(), clip_on=False)
                ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                        transform=ax.get_xaxis_transform(), clip_on=False)
                canvas = FigureCanvasTkAgg(fig, master=self.plots_frame)
                canvas.draw()
                canvas.get_tk_widget().grid(row=6, column=0, pady=20)

                # fig, ax = plt.subplots(figsize=(6, 4))
                # x_values = res.lambda2_list
                # y_values = res.result_wait_time_lambda2[0]
                # l, = ax.plot(x_values, y_values)
                # l.set_label(f'остальные факторы = {fixed}')
                # ax.legend()
                # ax.grid(True)
                # ax.set_xlabel("Интенсивность заявок типа 2, lambda2")
                # ax.set_ylabel("Среднее время ожидания типа 1, t1")
                # # Arrows
                # ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
                #         transform=ax.get_yaxis_transform(), clip_on=False)
                # ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                #         transform=ax.get_xaxis_transform(), clip_on=False)
                # canvas = FigureCanvasTkAgg(fig, master=self.plots_frame)
                # canvas.draw()
                # canvas.get_tk_widget().grid(row=7, column=0, pady=20)

                fig, ax = plt.subplots(figsize=(6, 4))
                x_values = res.lambda1_list
                y_values = res.result_wait_time_lambda1[1]
                l, = ax.plot(x_values, y_values)
                l.set_label(f'остальные факторы = {fixed}')
                ax.legend()
                ax.grid(True)
                ax.set_xlabel("Интенсивность заявок типа 1, lambda1")
                ax.set_ylabel("Среднее время ожидания типа 2, t2")
                # Arrows
                ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
                        transform=ax.get_yaxis_transform(), clip_on=False)
                ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                        transform=ax.get_xaxis_transform(), clip_on=False)
                canvas = FigureCanvasTkAgg(fig, master=self.plots_frame)
                canvas.draw()
                canvas.get_tk_widget().grid(row=8, column=0, pady=20)

                # fig, ax = plt.subplots(figsize=(6, 4))
                # x_values = res.lambda2_list
                # y_values = res.result_wait_time_lambda2[1]
                # l, = ax.plot(x_values, y_values)
                # l.set_label(f'остальные факторы = {fixed}')
                # ax.legend()
                # ax.grid(True)
                # ax.set_xlabel("Интенсивность заявок типа 2, lambda2")
                # ax.set_ylabel("Среднее время ожидания типа 2, t2")
                # # Arrows
                # ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
                #         transform=ax.get_yaxis_transform(), clip_on=False)
                # ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                #         transform=ax.get_xaxis_transform(), clip_on=False)
                # canvas = FigureCanvasTkAgg(fig, master=self.plots_frame)
                # canvas.draw()
                # canvas.get_tk_widget().grid(row=9, column=0, pady=20)

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
