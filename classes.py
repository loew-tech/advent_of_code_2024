import math
import re
from typing import List

from constants import CARDINAL_DIRECTIONS


class PatrolGuard:

    def __init__(self, grid: List[str | List[str]]):
        self._grid = grid
        for y, row in enumerate(grid):
            for x, v in enumerate(row):
                if v == '^':
                    self._y, self._x = y, x
                    break
        self._incs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self._i = 0

    def get_does_patrol_loop_and_patrol_area(self):
        def inbounds(y, x: int) -> bool:
            return 0 <= y < len(self._grid) and 0 <= x < len(self._grid[y])

        spaces, visited = set(), set()
        while inbounds(self._y, self._x):
            yi, xi = self._incs[self._i]
            if (self._y, self._x, yi, xi) in visited:
                return True, spaces
            visited.add((self._y, self._x, yi, xi))
            spaces.add((self._y, self._x))

            if not inbounds(self._y + yi, self._x + xi) or \
                    not self._grid[self._y + yi][self._x + xi] == '#':
                self._y += yi
                self._x += xi
            else:
                self._i = (self._i + 1) % len(self._incs)

        return False, spaces


class LinearSystem:

    def __init__(self, entry, part: str):
        entries = entry.split('\n')
        self._part = part
        self.ax, self.ay = [int(i) for i in re.findall(r'\d+', entries[0])]
        self.bx, self.by = [int(i) for i in re.findall(r'\d+', entries[1])]
        self.x, self.y = [int(i) for i in re.findall(r'\d+', entries[2])]
        self.x += 10_000_000_000_000 * (not self._part.upper() == 'A')
        self.y += 10_000_000_000_000 * (not self._part.upper() == 'A')

    def solve(self):
        ratio = (self.ax / self.ay)
        b = (self.x - ratio * self.y) / (self.bx - ratio * self.by)
        a = (self.x - self.bx * b) / self.ax
        return a, b

    def get_prize(self):
        limit = 100 if self._part.upper() == 'A' else float('inf')
        a, b = self.solve()
        if 0.999 <= a % 1 <= 0.99999999999999999999999999999999999999:
            a = math.ceil(a)
        if 0.999 <= b % 1 <= 0.99999999999999999999999999999999999999:
            b = math.ceil(b)

        attainable = min(a, b) >= 0 and \
                     max(a % 1, b % 1) < .01 and \
                     max(a, b) <= limit
        return attainable, int(3 * a + b)


class SecurityRobot:

    def __init__(self, robot: List[int], width=101, height=103):
        self.x, self.y, self._xi, self._yi = robot
        self._width, self._height = width, height

    def move(self):
        self.x = (self.x + self._xi) % self._width
        self.y = (self.y + self._yi) % self._height
        return self.y, self.x


class WarehouseRobot:

    def __init__(self, map_: List[str], moves: str):
        self._map, self._moves = map_, moves
        self._boxes = set()
        for y, row in enumerate(map_):
            for x, char in enumerate(row):
                if char == '@':
                    self._y, self._x = y, x
                elif char == 'O':
                    self._boxes.add((y, x))

    def move(self) -> None:
        movements = dict(zip('^<>v', CARDINAL_DIRECTIONS))

        for dir_ in self._moves:
            yi, xi = movements[dir_]
            y, x = self._y + yi, self._x + xi
            y_, x_, = y, x
            while (y_, x_) in self._boxes:
                y_ += yi
                x_ += xi
            if self._map[y_][x_] == '#':
                continue
            self._y, self._x = y, x
            if not (y, x) == (y_, x_):
                self._boxes.remove((y, x))
                self._boxes.add((y_, x_))

    def calc_gps_sum(self) -> int:
        return sum(y * 100 + x for y, x in self._boxes)


class WarehouseRobotB:

    def __init__(self, map_: List[str], moves: str):
        self._doubles = set()
        map_ = [[char for char in line for _ in range(2)] for line in map_]
        for y, row in enumerate(map_):
            skip = False
            for x, char in enumerate(row):
                if skip:
                    skip = False
                elif char == 'O':
                    self._doubles.add((y, x, x + 1))
                    skip = True
                elif char == '@':
                    self._y, self._x = y, x
        self._map, self._moves = map_, moves
        self._doubles_count = len(self._doubles)

    def move(self) -> None:
        movements = dict(zip('^<>v', CARDINAL_DIRECTIONS))

        for dir_ in self._moves:
            yi, xi = movements[dir_]
            y, x = self._y + yi, self._x + xi
            if self._map[y][x] == '#':
                continue
            if not yi:
                self._move_x(y, x, xi)
            else:
                self._move_y(y, x, yi)

    def _move_x(self, y, x, xi: int) -> None:
        y_, x_, = y, x
        to_move, (left_offset, right_offset) = [], (-1, 0) if xi == -1 else (
            0, 1)
        while (y_, x_ + left_offset, x_ + right_offset) in self._doubles:
            to_move.append((y_, x_ + left_offset, x_ + right_offset))
            if self._map[y_][x_ + xi] == '#':
                to_move = []
                break
            x_ += 2 * xi
        if self._map[y_][x_] == '#':
            return
        for (yy, xx, xxx) in to_move:
            self._doubles.remove((yy, xx, xxx))
            self._doubles.add((yy, xx + xi, xxx + xi))
        self._y, self._x = y, x

    def _move_y(self, y, x, yi: int) -> None:
        y_, x_, = y, x
        to_search = [(y_, x_ - 1, x_)] if \
            (y_, x_ - 1, x_) in self._doubles else []
        to_search = to_search or ([(y_, x_, x_ + 1)] if
                                  (y_, x_,
                                   x_ + 1) in self._doubles else [])
        to_move = []
        blocked = False
        while to_search:
            next_search = set()
            for yy, x_left, x_right in to_search:
                if '#' in self._map[yy + yi][x_left] == '#' or \
                        self._map[yy + yi][x_right] == '#':
                    blocked = True
                    to_move = []
                    next_search = []
                    break
                to_move.append((yy, x_left, x_right))
                if (yy + yi, x_left + 1, x_right + 1) in self._doubles:
                    next_search.add((yy + yi, x_left + 1, x_right + 1))
                if (yy + yi, x_left - 1, x_right - 1) in self._doubles:
                    next_search.add((yy + yi, x_left - 1, x_right - 1))
                if (yy + yi, x_left, x_right) in self._doubles:
                    next_search.add((yy + yi, x_left, x_right))
            to_search = next_search

        self._y, self._x = (y, x) if not blocked else (self._y, self._x)

        added = set()
        for box in to_move:
            if box not in added:
                self._doubles.remove(box)
            yy, x_left, x_right = box
            added.add((yy + yi, x_left, x_right))
            self._doubles.add((yy + yi, x_left, x_right))

    def calc_gps_sum(self) -> int:
        return sum(y * 100 + x for y, x, _ in self._doubles)


class Computer:

    def __init__(self, ar, br, cr: int):

        self._regs = [ar, br, cr]
        a, b, c = 0, 1, 2
        self._i = 0
        self._get_operand = lambda x: {4: self._regs[a], 5: self._regs[b],
                                       6: self._regs[c]}.get(x, x)
        self._ops = {
            0: lambda x: (a, self._regs[a] // 2**self._get_operand(x)),
            1: lambda x: (b, self._regs[b] ^ x),
            2: lambda x: (b, self._get_operand(x) % 8),
            3: lambda x: (None, x if self._regs[a] else -1),
            4: lambda _: (b, self._regs[b] ^ self._regs[c]),
            5: lambda x: (False, self._get_operand(x) % 8),
            6: lambda x: (b, self._regs[a] // 2**self._get_operand(x)),
            7: lambda x: (c, self._regs[a] // 2**self._get_operand(x)),
        }

    def execute(self, program: List[int]) -> List[int]:
        output = []
        while self._i < len(program):
            reg, val = self._ops[program[self._i]](program[self._i + 1])
            if reg in {0, 1, 2} and reg is not False:
                self._regs[reg] = val
            elif reg is not None:
                output.append(val)
            elif val >= 0:
                self._i = val
                continue
            self._i += 2
        return output

    def reset(self, a, b, c: int) -> None:
        self._regs, self._i = [a, b, c], 0
