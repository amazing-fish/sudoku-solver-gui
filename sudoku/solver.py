def is_valid(puzzle, row, col, num):
    # Check if the number is in the same row or column
    for i in range(9):
        if puzzle[row][i] == num or puzzle[i][col] == num:
            return False

    # Check if the number is in the same 3x3 box
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if puzzle[start_row + i][start_col + j] == num:
                return False

    return True


def solve_sudoku(puzzle):
    empty_cell = find_empty_cell(puzzle)
    if not empty_cell:
        return True  # The puzzle is solved

    row, col = empty_cell

    for num in range(1, 10):
        if is_valid(puzzle, row, col, num):
            puzzle[row][col] = num

            if solve_sudoku(puzzle):
                return True

            # Backtrack
            puzzle[row][col] = 0

    return False  # No solution found


def find_empty_cell(puzzle):
    for row in range(9):
        for col in range(9):
            if puzzle[row][col] == 0:
                return row, col
    return None
