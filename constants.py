DIRECTIONS = tuple((i, j) for i in range(-1, 2)
                   for j in range(-1, 2) if not i == j == 0)


CARDINAL_DIRECTIONS = tuple((i, j) for i in range(-1, 2)
                            for j in range(-1, 2) if not abs(i) == abs(j))
