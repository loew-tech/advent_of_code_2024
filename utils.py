from collections import Counter, defaultdict
from datetime import datetime
from functools import cache
from http import HTTPStatus
import re
import requests
from typing import List, Callable, Tuple, Set, Dict

from sortedcontainers import SortedList

from classes import SecurityRobot
from constants import DIRECTIONS, CARDINAL_DIRECTIONS

ADVENT_URI = 'https://adventofcode.com/'


def read_input(day: int | str, delim='\n', year=None) -> List[str] | str:
    year = year if year is not None else datetime.now().year
    with open('.env') as env_:
        session_id = env_.read().strip().split('\n')[0]
    response = requests.get(f'{ADVENT_URI}{year}/day/{day}/input',
                            cookies={'session': session_id})
    if response.status_code == HTTPStatus.OK:
        return response.text.strip().split(delim) if delim else response.text


def get_inbounds(grid: List[List[any] | str]) -> Callable[[int, int], bool]:
    return lambda y, x: inbounds(y, x, grid)


def inbounds(y, x: int, grid: List[List[any] | str]) -> bool:
    return 0 <= y < len(grid) and 0 <= x < len(grid[y])


def get_grid_stop_start(grid: [List[str | List[str]]]) -> \
        Tuple[Tuple[int, int], Tuple[int, int]]:
    start_y, start_x, end_y, end_x = None, None, None, None
    for y_, row in enumerate(grid):
        for x_, v in enumerate(row):
            if v == 'S':
                start_y, start_x = y_, x_
            elif v == 'E':
                end_y, end_x = y_, x_
    return (start_y, start_x), (end_y, end_x)


def day_2_helper(part='A') -> int:
    def _is_valid_report(report_: List[str]) -> int:
        diffs = {int(v) - int(report_[i_]) for i_, v in enumerate(report_[1:])}
        return diffs <= {1, 2, 3} or diffs <= {-1, -2, -3}

    count = 0
    for report in (r.split() for r in read_input(2)):
        if part.upper() == 'A':
            count += _is_valid_report(report)
        else:
            for i in range(len(report)):
                if _is_valid_report(report[:i] + report[i + 1:]):
                    count += 1
                    break
    return count


def day_3_sum_mult(data: str) -> int:
    pattern = r'mul\(\d+,\d+\)'
    matches = re.findall(pattern, data)
    sum_ = 0
    for m in matches:
        d1, d2 = re.findall(r'\d+', m)
        sum_ += int(d1) * int(d2)
    return sum_


def day_4_word_search(data: List[str], part: str) -> int:
    count, xmas = 0, 'XMAS'
    is_inbounds = get_inbounds(data)

    def is_xmas(y_, x_, yi_, xi_: int) -> bool:
        i = 0
        while is_inbounds(y_, x_, ) and i < len(xmas) and \
                data[y_][x_] == xmas[i]:
            y_ += yi_
            x_ += xi_
            i += 1
        return i == len(xmas)

    def is_mas(y_, x_: int) -> bool:
        if not data[y_][x_] == 'A':
            return False
        if not (is_inbounds(y_ - 1, x_ - 1) and is_inbounds(y + 1, x_ + 1) and
                {data[y_ - 1][x_ - 1], data[y_ + 1][x_ + 1]} == {'S', 'M'}):
            return False
        return is_inbounds(y_ - 1, x_ + 1) and is_inbounds(y_ + 1, x_ - 1) and\
               {data[y_ - 1][x_ + 1], data[y_ + 1][x_ - 1]} == {'S', 'M'}

    for y, row in enumerate(data):
        for x in range(len(row)):
            count += sum(is_xmas(y, x, yi, xi) for yi, xi in DIRECTIONS) if \
                part.upper() == 'A' else is_mas(y, x)
    return count


def day_5_sum_mid_page(predecessors: defaultdict,
                       updates: List[List[int]], part='A') -> int:
    def is_ordered(update_: List[int]) -> bool:
        observed = set()
        for page in update_:
            if observed - predecessors[page]:
                return False
            observed.add(page)
        return True

    def fix_update(update_: List[int]) -> List[int]:
        vals, reuse, fixed_ = set(update_), set(), []
        while vals:
            val = vals.pop()
            if vals & predecessors[val]:
                reuse.add(val)
            else:
                fixed_.append(val)
                vals.update(reuse)
                reuse = set()
        return fixed_

    sum_ = 0
    for update in updates:
        if is_ordered(update):
            sum_ += (part.upper() == 'A') * update[len(update) // 2]
            continue
        if part.upper() == 'A':
            continue
        sum_ += fix_update(update)[len(update) // 2]
    return sum_


def day_7_check_eq(sol: int, operands: List[int], part='A') -> bool:
    def check(i, temp: int) -> bool:
        if i == len(operands):
            return temp == sol
        return check(i + 1, temp + operands[i]) or \
               check(i + 1, temp * operands[i]) or (
                       not part.upper() == 'A' and
                       check(i + 1, int(f'{temp}{operands[i]}')))

    return check(1, operands[0])


def day_8_count_antinodes(antennas: defaultdict,
                          inbounds_: Callable[[int, int], bool],
                          part='A') -> int:
    antinodes = set()
    for antennas_ in antennas.values():
        for y, x in antennas_:
            for y1, x1 in antennas_:
                if y == y1 and x == x1:
                    continue

                dy, dx = y - y1, x - x1
                antinodes_y, antinodes_x = y1 - dy, x1 - dx
                if inbounds_(antinodes_y, antinodes_x):
                    antinodes.add((antinodes_y, antinodes_x))

                y_, x_ = y1, x1
                while not part.upper() == 'A' and inbounds_(y_, x_):
                    antinodes.add((y_, x_))
                    y_ -= dy
                    x_ -= dx
    return len(antinodes)


def day_9_compress_map(data: List[int]) -> int:
    start, end, map_ = 0, len(data) - 2 + len(data) % 2, []
    while start < end:
        map_.append((start // 2, data[start]))
        data[start] = 0
        start += 1
        while start < end and data[end] <= data[start]:
            data[start] -= data[end]
            map_.append((end // 2, data[end]))
            data[end] = 0
            end -= 2
        if start < end and 0 < data[start]:
            data[end] -= data[start]
            map_.append((end // 2, data[start]))
        start += 1

    for v in end, start:
        if data[v]:
            map_.append((v // 2, data[v]))
            data[v] = 0

    i = -1
    return sum(id_ * (i := i + 1) for id_, len_ in map_
               for _ in range(len_))


def day_9b_compress_map(data: List[int]) -> int:
    start, end, map_, modified = 0, len(data) - 2 + len(data) % 2, [], True
    used = set()
    while start <= end:
        if start in used:
            map_.append((0, data[start]))
        else:
            map_.append((start // 2, data[start]))
        data[start] = 0
        start += 1
        if start == len(data):
            break
        i = end + 2
        while data[start] and (i := i - 2) > start:
            if i in used:
                continue
            if data[i] and data[i] <= data[start]:
                map_.append((i // 2, data[i]))
                data[start] -= data[i]
                used.add(i)
        map_.append((0, data[start]))
        data[start] = 0
        start += 1

    i = -1
    return sum(id_ * (i := i + 1) for id_, len_ in map_
               for _ in range(len_))


def day_10_sum_scores(data: List[List[int]], part='A') -> int:
    starts = [(y, x) for y, row in enumerate(data) for x, v in enumerate(row)
              if not v]
    inbounds_ = get_inbounds(data)

    def bfs(start_: Tuple[int, int]) -> int:
        current_val = 0
        to_search = {start_: 1}
        while to_search and current_val < 9:
            current_val += 1
            next_search = defaultdict(int)
            for (y, x), paths in to_search.items():
                for yi, xi in CARDINAL_DIRECTIONS:
                    y_, x_ = y + yi, x + xi
                    if inbounds_(y_, x_) and data[y_][x_] == current_val:
                        next_search[(y_, x_)] += paths
            to_search = next_search
        return current_val == 9 and (part.upper() == 'A' and len(to_search)
                                     or sum(to_search.values()))

    return sum(bfs(start) for start in starts)


def day_11_blink_stones(data: List[int], iterations: int) -> int:
    @cache
    def blink(stone_) -> List[int]:
        if not stone_:
            return [1]
        elif not len(str_ := str(stone_)) % 2:
            mid_ = len(str_) // 2
            return [int(str_[:mid_]), int(str_[mid_:])]
        else:
            return [stone_ * 2024]

    counter = Counter(data)
    for _ in range(iterations):
        new_counts = defaultdict(int)
        for stone, count in counter.items():
            for new_stone in blink(stone):
                new_counts[new_stone] += count
        counter = new_counts
    return sum(counter.values())


def day_12_calc_fence_cost(data: List[str], part) -> int:
    inbounds_, used = get_inbounds(data), set()

    def bfs(y_, x_: int, target: str) -> Tuple[Set, Set, int]:
        to_search, perim, area, perim_size_ = [(y_, x_)], set(), set(), 0
        while to_search:
            next_search = set()
            for yi, xi in to_search:
                if (yi, xi) in area:
                    continue
                area.add((yi, xi))
                used.add((yi, xi))
                for yj, xj in CARDINAL_DIRECTIONS:
                    if inbounds_(yi + yj, xi + xj) and \
                            data[yi + yj][xi + xj] == target:
                        next_search.add((yi + yj, xi + xj))
                    else:
                        perim_size_ += 1
                        perim.add((yi, xi))
            to_search = next_search
        return perim, area, perim_size_

    def calc_nums_sides(perim: Set[Tuple[int, int]],
                        target: str) -> int:
        y_lines, x_lines = defaultdict(SortedList), defaultdict(SortedList)
        for y_, x_ in perim:
            y_lines[y_].add(x_)
            x_lines[x_].add(y_)

        sides = 0
        for y_, lst in y_lines.items():
            last_u, last_d = -2, -2
            for x_ in lst:
                if not inbounds_(y_ - 1, x_) or not data[y_ - 1][x_] == target:
                    sides += not (x_ == last_u + 1)
                    last_u = x_
                if not inbounds_(y_ + 1, x_) or not data[y_ + 1][x_] == target:
                    sides += not (x_ == last_d + 1)
                    last_d = x_

        for x_, lst in x_lines.items():
            last_l, last_r = -2, -2
            for y_ in lst:
                if not inbounds_(y_, x_ - 1) or not data[y_][x_ - 1] == target:
                    sides += not (y_ == last_l + 1)
                    last_l = y_
                if not inbounds_(y_, x_ + 1) or not data[y_][x_ + 1] == target:
                    sides += not (y_ == last_r + 1)
                    last_r = y_
        return sides

    cost = 0
    for y, row in enumerate(data):
        for x, v in enumerate(row):
            if (y, x) in used:
                continue
            perim_, area_, perim_size = bfs(y, x, v)
            if part.upper() == 'A':
                cost += perim_size * len(area_)
            else:
                cost += calc_nums_sides(perim_, v) * len(area_)
    return cost


def day_14_calc_quadrant_prod(data: List[List[int]]) -> int:
    q1, q2, q3, q4 = 0, 0, 0, 0

    def inc_quads(y_, x_):
        nonlocal q1, q2, q3, q4
        if x_ < 50 and y_ < 51:
            q1 += 1
        elif x_ > 50 and y_ < 51:
            q2 += 1
        elif x_ < 50 and y_ > 51:
            q3 += 1
        elif x_ > 50 and y_ > 51:
            q4 += 1

    for x, y, xi, yi in data:
        x = (x + xi * 100) % 101
        x = x if x >= 0 else 100 + x

        y = (y + yi * 100) % 103
        y = y if y >= 0 else y + 102

        inc_quads(y, x)
    return q1 * q2 * q3 * q4


def day_14_find_tree(data: List[List[int]]) -> int:
    secs, num_bots, positions = 0, len(data), set()
    robots = [SecurityRobot(robot) for robot in data]
    while len(positions) < num_bots and (secs := secs + 1):
        positions = {bot.move() for bot in robots}
    return secs


def day_16_maze_costs(maze: List[str]) -> Tuple:
    (start_y, start_x), (end_y, end_x) = get_grid_stop_start(maze)
    costs = {(start_y, start_x, 0): 0}
    increments = ((0, 1), (1, 0), (0, -1), (-1, 0))

    to_search, min_ = {(start_y, start_x, 0)}, float('inf')
    while to_search:
        next_search = set()
        for y, x, i in to_search:
            if maze[y][x] == '#':
                del costs[(y, x, i)]
                continue
            if maze[y][x] == 'E':
                min_ = min(min_, costs[(y, x, i)])
                continue

            left, right = (i + 1) % 4, i - 1 if i else 3
            for turn in left, right:
                if (y, x, turn) not in costs or \
                        costs[(y, x, i)] + 1000 < costs[(y, x, turn)]:
                    costs[(y, x, turn)] = costs[(y, x, i)] + 1000
                    next_search.add((y, x, turn))

            yi, xi = increments[i]
            if (y + yi, x + xi, i) not in costs or \
                    costs[(y, x, i)] + 1 < costs[(y + yi, x + xi, i)]:
                costs[(y + yi, x + xi, i)] = costs[(y, x, i)] + 1
                next_search.add((y + yi, x + xi, i))
        to_search = next_search
    return (end_y, end_x), min_, costs


def day_16b_count_best_seats(ending_loc: Tuple, costs: Dict,
                             min_: int) -> int:
    increments = ((0, 1), (1, 0), (0, -1), (-1, 0))
    best_seats = {ending_loc}
    to_search = {(*ending_loc, i, min_) for i in range(4) if
                 costs.get((*ending_loc, i)) == min_}

    while to_search:
        next_search = set()
        for y, x, i, cost in to_search:
            min_, mins = float('inf'), set()
            yi, xi = increments[i]
            if (y - yi, x - xi, i) in costs and \
                    costs[(y - yi, x - xi, i)] < cost:
                min_ = costs[(y - yi, x - xi, i)]
                next_search.add((y - yi, x - xi, i, min_))

            left, right = (i + 1) % 4, i - 1 if i else 3
            for turn in left, right:
                if (y, x, turn) in costs and \
                        costs[(y, x, turn)] <= min(min_, costs[(y, x, i)]):
                    min_, mins = costs[(y, x, turn)], {(y, x, turn)}
            next_search |= {(y, x, i, min_) for y, x, i in mins}

        best_seats |= {(y, x) for y, x, *_ in next_search}
        to_search = next_search

    return len(best_seats)


def day_19_falling_memory(corrupted: Set[Tuple[int, int]]) -> int:
    end = (70, 70)

    def inbounds_(y_, x_) -> bool:
        return 0 <= y_ < 71 and 0 <= x_ < 71

    to_search, visited, count = {(0, 0)}, set(), 0
    while to_search and (count := count + 1):
        next_search = set()
        for y, x in to_search:
            visited.add((y, x))
            for yi, xi in CARDINAL_DIRECTIONS:
                if inbounds_(y + yi, x + xi) and (
                        y + yi, x + xi) not in corrupted \
                        and (y + yi, x + xi) not in visited:
                    if (y + yi, x + xi) == end:
                        return count
                    next_search.add((y + yi, x + xi))
        to_search = next_search
    return -1


def day_19_count_patterns(towels: defaultdict,
                          patterns: List[str],
                          part='A') -> int:
    @cache
    def check(pattern) -> int:
        if not pattern:
            return 1
        return sum(check(pattern[len(towel):]) for towel in
                   towels[pattern[0]] if pattern.startswith(towel))

    return sum(bool(check(p)) if part.upper() == 'A' else
               check(p) for p in patterns)


def day_22_gen_secrets(data: List[int]):
    def transform(secret_: int) -> int:
        temp = secret_ * 64
        secret_ ^= temp
        secret_ %= 16777216
        temp = secret_ // 32
        secret_ ^= temp
        secret_ %= 16777216
        temp = secret_ * 2048
        secret_ ^= temp
        secret_ %= 16777216
        return secret_

    sequences = defaultdict(int)
    for j in range(len(data)):
        str_, used = '', set()
        for i in range(2_000):
            last = data[j] % 10
            data[j] = transform(data[j])
            price = data[j] % 10
            delta = price - last
            if i < 3:
                str_ = f'{str_},{delta}'
            else:
                str_ = ','.join(str_.split(',')[1:] + [f'{delta}'])

            if i >= 3 and str_ not in used:
                sequences[str_] += price
                used.add(str_)
    return sum(data), max(sequences.values())


def day_23_get_graph_and_possibilities(data: List[List[str]]) \
        -> Tuple[defaultdict, Set[str]]:
    graph = defaultdict(set)
    possibilities = set()
    for cpu1, cpu2 in data:
        graph[cpu1].add(cpu2)
        graph[cpu2].add(cpu1)
        if cpu1[0] == 't':
            possibilities.add(cpu1)
        if cpu2[0] == 't':
            possibilities.add(cpu2)

    return graph, possibilities


def day_23_count_cycles(graph: defaultdict, possibilities: Set[str]) -> int:
    cycles = set()
    while possibilities:
        start = possibilities.pop()
        to_search, observed, depth = {(None, start)}, set(), 0
        while to_search and (depth := depth + 1) <= 3:
            next_search = set()
            for last, computer in to_search:
                observed.add(computer)
                for c in graph[computer]:
                    if c == start and not last == start:
                        cycles.add('-'.join(sorted((start, last, computer))))
                        continue
                    next_search.add((computer, c))
            to_search = next_search

    return len(cycles)


def get_max_network(graph: defaultdict) -> str:
    max_, comp = 0, {}
    for node in graph:
        start = graph[node] | {node}
        for n in graph[node]:
            if n not in start:
                continue
            start -= (start - (graph[n] | {n}))

        if max_ < len(start):
            max_ = len(start)
            comp = start

    return ','.join(sorted(comp))


def day_24_solve_gates(endz: Set[str], gates, vals: Dict, ops: Dict):
    wrong = set()
    highest_z = max(endz)
    xor, and_, or_ = "XOR", "AND", 'OR'

    def solve(val):
        op, wire1, wire2 = gates[val]
        if val[0] == 'z' and not op == ops[xor] and not val == highest_z:
            wrong.add(val)
        if val[0] not in 'xyz' and wire1[0] not in 'xyz' and \
                wire2[0] not in 'xyz' and op == ops[xor]:
            wrong.add(val)
        if op == ops[and_] and 'x00' not in (wire1, wire2):
            for res, (op_, w1, w2) in gates.items():
                if val in (w1, w2) and not op_ == ops[or_]:
                    wrong.add(val)
        if op == ops[xor]:
            for res, (op_, w1, w2) in gates.items():
                if val in (w1, w2) and op_ == ops[or_]:
                    wrong.add(val)

        if wire1 not in vals:
            solve(wire1)
        if wire2 not in vals:
            solve(wire2)
        result = op(vals[wire1], vals[wire2])
        vals[val] = result
        return result

    bits = [solve(z) for z in sorted(endz, reverse=True)]
    return int(''.join([str(bit) for bit in bits]), 2), ','.join(sorted(wrong))


def day_25_count_matches(schematics: List[List[str]]) -> int:
    locks, keys, height = [], [], len(schematics[0]) - 2
    for schematic in schematics:
        char_ = schematic[0][0]
        scheme = []
        for x in range(len(schematic[0])):
            y = 0
            while (y := y + 1) < len(schematic) and schematic[y][x] == char_:
                pass
            scheme.append(y-1)
        if char_ == '#':
            locks.append(scheme)
        else:
            keys.append([height - i for i in scheme])

    count = 0
    for key in keys:
        for lock in locks:
            valid = True
            print(f'{key=} {lock=}')
            for i, v in enumerate(key):
                if v + lock[i] > height:
                    print(f'\tcollision at column {i+1}')
                    valid = False
                    break
            count += valid

    return count
