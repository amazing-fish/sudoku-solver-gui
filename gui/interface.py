import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

from sudoku.solver import solve_sudoku
from sudoku.validator import is_valid_input

from image_processing.image_to_puzzle import recognize_sudoku_puzzle


class SudokuGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("数独求解器")
        self.resizable(0, 0)

        self.original_puzzle = [[0 for _ in range(9)] for _ in range(9)]

        self.board_frame = tk.Frame(self, padx=10, pady=10)
        self.board_frame.pack()

        self.entries = [
            [
                tk.Entry(self.board_frame, width=3, font=("Arial", 14), justify="center")
                for _ in range(9)
            ]
            for _ in range(9)
        ]

        for i, row in enumerate(self.entries):
            for j, entry in enumerate(row):
                entry.grid(
                    row=i,
                    column=j,
                    padx=(1 if j % 3 != 2 else 4),
                    pady=(1 if i % 3 != 2 else 4),
                )

        self.control_frame = tk.Frame(self, pady=10)
        self.control_frame.pack()

        self.solve_button = tk.Button(self.control_frame, text="求解", command=self.solve, width=10)
        self.solve_button.grid(row=0, column=0, padx=5)

        self.reset_button = tk.Button(self.control_frame, text="重置", command=self.reset, width=10)
        self.reset_button.grid(row=0, column=1, padx=5)

        self.load_image_button = tk.Button(self.control_frame, text="上传图片", command=self.load_image, width=10)
        self.load_image_button.grid(row=0, column=2, padx=5)

    def load_image(self):
        image_file = filedialog.askopenfilename(
            title="选择包含数独的图片",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")],
        )
        if not image_file:
            return

        try:
            puzzle = recognize_sudoku_puzzle(image_file)
        except FileNotFoundError as exc:
            messagebox.showerror("模型缺失", str(exc))
            return
        except Exception as exc:  # pylint: disable=broad-except
            messagebox.showerror("识别失败", f"识别图片时出现问题：{exc}")
            return

        self.original_puzzle = [row.copy() for row in puzzle]
        self.set_puzzle(puzzle)

    def get_puzzle(self):
        puzzle = []
        for row in self.entries:
            current_row = []
            for entry in row:
                value = entry.get()
                current_row.append(int(value) if value.isdigit() else 0)
            puzzle.append(current_row)

        self.original_puzzle = [row.copy() for row in puzzle]
        return puzzle

    def set_puzzle(self, puzzle):
        for i, row in enumerate(puzzle):
            for j, num in enumerate(row):
                entry = self.entries[i][j]
                entry.delete(0, tk.END)
                if num != 0:
                    entry.insert(0, str(num))

                if num != 0 and self.original_puzzle[i][j] == num:
                    entry.config(fg="black")
                elif num != 0:
                    entry.config(fg="steel blue")
                else:
                    entry.config(fg="black")

    def solve(self):
        puzzle = self.get_puzzle()

        if not is_valid_input(puzzle):
            messagebox.showerror("输入无效", "请输入有效的数独题目。")
            return

        if solve_sudoku(puzzle):
            self.set_puzzle(puzzle)
        else:
            messagebox.showinfo("无解", "未找到该数独题目的解。")

    def reset(self):
        for row in self.entries:
            for entry in row:
                entry.delete(0, tk.END)
                entry.config(fg="black")

        self.original_puzzle = [[0 for _ in range(9)] for _ in range(9)]
