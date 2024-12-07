from collections import Counter, defaultdict
import inspect
import sys

from classes import PatrolGuard
from utils import (read_input, day_2_helper, day_3_sum_mult, day_4_word_search,
                   day_5_sum_mid_page, day_7_check_eq)


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
