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

        def inbounds(y_loc, x_loc: int) -> bool:
            return 0 <= y_loc < len(self._map) and\
                   0 <= x_loc < len(self._map[y_loc])

        for dir_ in self._moves:
            yi, xi = movements[dir_]
            y, x = self._y + yi, self._x + xi
            y_, x_, moves = y, x, []
            while inbounds(y_, x_) and (y_, x_) in self._boxes:
                moves.append((y_, x_))
                y_ += yi
                x_ += xi
            if inbounds(y_, x_) and self._map[y_][x_] == '#':
                continue
            self._y, self._x = y, x
            if moves:
                self._boxes.remove(moves[0])
                self._boxes.add((y_, x_))

    def calc_gps_sum(self) -> int:
        return sum(y * 100 + x for y, x in self._boxes)
