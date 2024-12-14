import re
from collections import Counter, defaultdict
import inspect
import sys

from classes import PatrolGuard, LinearSystem
from utils import (read_input, get_inbounds, day_2_helper, day_3_sum_mult,
                   day_4_word_search, day_5_sum_mid_page, day_7_check_eq,
                   day_8_count_antinodes, day_9_compress_map,
                   day_9b_compress_map, day_10_sum_scores, day_11_blink_stones,
                   day_12_calc_fence_cost, day_14_calc_quadrant_prod,
                   day_14_find_tree)


def day_1(part='A') -> int:
    data = read_input(1)
    first = sorted(int(i.split()[0]) for i in data)
    second = (int(i.split()[1]) for i in data)
    if part.upper() == 'A':
        second = sorted(second)
        return sum(abs(v - second[i]) for i, v in enumerate(first))
    counter = Counter(second)
    return sum(v * counter.get(v, 0) for v in first)


def day_2(part='A') -> int:
    return day_2_helper(part)


def day_3(part='A') -> int:
    data = read_input(3, delim=None)
    if part.upper() == 'A':
        return day_3_sum_mult(data)
    return sum(day_3_sum_mult(entry) for entry in data.split('do')
               if not entry.startswith("n't"))


def day_4(part='A') -> int:
    return day_4_word_search(read_input(4), part)


def day_5(part='A') -> int:
    data = read_input(5, delim=None).split('\n\n')
    data, updates = data
    data = [[int(i) for i in row.split('|')] for row in data.split('\n')]
    updates = [[int(i) for i in row.split(',')] for
               row in updates.split('\n')[:-1]]
    predecessors = defaultdict(set)
    for x, y in data:
        predecessors[y].add(x)
    return day_5_sum_mid_page(predecessors, updates, part)


def day_6(part='A') -> int:
    data = read_input(6)
    _, path = PatrolGuard(data).get_does_patrol_loop_and_patrol_area()
    if part.upper() == 'A':
        return len(path)
    data, count = [list(str_) for str_ in data], 0
    for y, x in path:
        if data[y][x] == '^':
            continue
        data[y][x] = '#'
        loops, _ = PatrolGuard(data).get_does_patrol_loop_and_patrol_area()
        count += loops
        data[y][x] = '.'
    return count


def day_7(part='A') -> int:
    data = [row.split(':') for row in read_input(7)]
    ret, invalids = 0, []
    for sol, operands in data:
        sol, operands = int(sol), [int(i) for i in operands.split()]
        if day_7_check_eq(sol, operands, part='A'):
            ret += sol
        else:
            invalids.append((sol, operands))
    return ret if part.upper() == 'A' else \
        ret + sum(sol for sol, ops in invalids if
                  day_7_check_eq(sol, ops, part='B'))


def day_8(part='A') -> int:
    data = read_input(8)
    inbounds, antennas = get_inbounds(data), defaultdict(list)

    for y, row in enumerate(data):
        for x, v in enumerate(row):
            if v == '.':
                continue
            antennas[v].append((y, x))

    return day_8_count_antinodes(antennas, inbounds, part)


def day_9(part='A') -> int:
    data = [int(i) for i in read_input(9, delim=None).strip()]
    if part.upper() == 'A':
        return day_9_compress_map(data)
    return day_9b_compress_map(data)


def day_10(part='A') -> int:
    return day_10_sum_scores([[int(i) for i in row] for row in read_input(10)],
                             part)


def day_11(part='A') -> int:
    data = [int(i) for i in read_input(11, delim=None).split()]
    iterations = 25 if part.upper() == 'A' else 75
    return day_11_blink_stones(data, iterations)


def day_12(part='A') -> int:
    return day_12_calc_fence_cost(read_input(12, delim='\n'), part)


def day_13(part='A') -> int:
    data, cost = read_input(13, delim='\n\n'), 0
    for entry in data:
        attainable, tokens = LinearSystem(entry, part).get_prize()
        cost += attainable * tokens
    return cost


def day_14(part='A') -> int:
    data = [[int(i) for i in re.findall(r'-?\d+', row)]
            for row in read_input(14)]
    if part.upper() == 'A':
        return day_14_calc_quadrant_prod(data)
    # @TODO: partB 6645 is too high
    return day_14_find_tree(data)


if __name__ == '__main__':
    args = sys.argv[1:] if sys.argv[1:] else range(1, 26)
    args = (f'day_{i}' for i in args)
    members = inspect.getmembers(inspect.getmodule(inspect.currentframe()))
    funcs = {name: member for name, member in members
             if inspect.isfunction(member)}
    for day in args:
        if day not in funcs:
            print(f'{day}()= NotImplemented')
            continue
        print(f'{day}()= {funcs[day]()}')
        print(f'{day}(part="B")= {funcs[day](part="B")}')
