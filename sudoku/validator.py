def is_valid_input(puzzle):
    for row in range(9):
        for col in range(9):
            num = puzzle[row][col]

            if num == 0:
                continue

            if not (1 <= num <= 9) or not isinstance(num, int):
                return False

            if not is_unique_in_row(puzzle, row, num) or not is_unique_in_col(puzzle, col, num) or not is_unique_in_box(puzzle, row, col, num):
                return False

    return True


def is_unique_in_row(puzzle, row, num):
    return puzzle[row].count(num) == 1


def is_unique_in_col(puzzle, col, num):
    return [puzzle[row][col] for row in range(9)].count(num) == 1


def is_unique_in_box(puzzle, row, col, num):
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)

    for i in range(3):
        for j in range(3):
            if (i != row % 3 or j != col % 3) and puzzle[start_row + i][start_col + j] == num:
                return False

    return True
