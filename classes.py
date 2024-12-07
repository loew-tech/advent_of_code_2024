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

    def get_does_patrol_loop_and_patrol_size(self):
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
