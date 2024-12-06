from typing import List


class PatrolGuard:

    def __init__(self, grid: List[str]):
        self._grid = grid
        for y, row in enumerate(grid):
            for x, v in enumerate(row):
                if v == '^':
                    self._y, self._x = y, x
                    break
        self._incs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self._i = 0

    def patrol(self):
        def inbounds(y, x: int) -> bool:
            return 0 <= y < len(self._grid) and 0 <= x < len(self._grid[y])

        visited = set()

        while inbounds(self._y, self._x):
            visited.add((self._y, self._x))
            yi, xi = self._incs[self._i]
            if not inbounds(self._y + yi, self._x + xi) or \
                    not self._grid[self._y + yi][self._x + xi] == '#':
                self._y += yi
                self._x += xi
            else:
                self._i = (self._i + 1) % len(self._incs)

        return len(visited)
