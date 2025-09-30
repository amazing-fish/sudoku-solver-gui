import tkinter as tk
from tkinter import messagebox
from sudoku.solver import solve_sudoku
from sudoku.validator import is_valid_input
from tkinter import filedialog
from image_processing.image_to_puzzle import recognize_sudoku_puzzle

class SudokuGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Sudoku Solver")
        self.geometry("370x310")
        self.resizable(0, 0)
        self.load_image_button = tk.Button(self, text="Load Image", command=self.load_image)
        self.load_image_button.grid(row=10, column=0, columnspan=9, pady=5)
        self.entries = [[tk.Entry(self, width=3, font=("Arial", 14), justify="center") for _ in range(9)] for _ in
                        range(9)]

        for i, row in enumerate(self.entries):
            for j, entry in enumerate(row):
                entry.grid(row=i, column=j, padx=(1 if j % 3 != 2 else 4), pady=(1 if i % 3 != 2 else 4))

        self.solve_button = tk.Button(self, text="Solve", command=self.solve)
        self.solve_button.grid(row=9, column=0, columnspan=4, pady=5)

        self.reset_button = tk.Button(self, text="Reset", command=self.reset)
        self.reset_button.grid(row=9, column=5, columnspan=4, pady=5)

    def load_image(self):
        image_file = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
        if not image_file:
            return

        puzzle = recognize_sudoku_puzzle(image_file)
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
                if num != 0:
                    self.entries[i][j].delete(0, tk.END)
                    self.entries[i][j].insert(0, str(num))

                    # Set the text color based on whether the cell has been changed
                    if self.original_puzzle[i][j] == num:
                        self.entries[i][j].config(fg="black")
                    else:
                        self.entries[i][j].config(fg="light blue")

    def solve(self):
        puzzle = self.get_puzzle()

        if not is_valid_input(puzzle):
            messagebox.showerror("Invalid Input", "Please enter a valid Sudoku puzzle.")
            return

        if solve_sudoku(puzzle):
            self.set_puzzle(puzzle)
        else:
            messagebox.showinfo("No Solution", "No solution found for this Sudoku puzzle.")

    def reset(self):
        for row in self.entries:
            for entry in row:
                entry.delete(0, tk.END)
                entry.config(fg="black")

