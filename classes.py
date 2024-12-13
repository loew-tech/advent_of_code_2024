import math
import re
from typing import List


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

    def __init__(self, entry: str):
        entries = entry.split('\n')
        self.ax, self.ay = [int(i) for i in re.findall(r'\d+', entries[0])]
        self.bx, self.by = [int(i) for i in re.findall(r'\d+', entries[1])]
        self.x, self.y = [int(i) for i in re.findall(r'\d+', entries[2])]

    def solve(self):
        ratio = (self.ax / self.ay)
        b = (self.x - ratio * self.y) / (self.bx - ratio * self.by)
        a = (self.x - self.bx * b) / self.ax
        return a, b

    def get_prize(self, limit=100):
        a, b = self.solve()
        if 0.999 <= a % 1 <= 0.99999999999999999999999999999999999999:
            a = math.ceil(a)
        if 0.999 <= b % 1 <= 0.99999999999999999999999999999999999999:
            b = math.ceil(b)

        attainable = min(a, b) >= 0 and \
                     max(a % 1, b % 1) < .0001 and \
                     max(a, b) <= limit
        return attainable, int(3 * a + b)

    def __repr__(self):
        return f'ax={self.ax} ay={self.ay} bx={self.bx} by={self.by} x=' \
               f'{self.x} y={self.y}'
