from collections import Counter, defaultdict
import inspect
import sys

from utils import (read_input, day_2_helper, day_3_sum_mult, day_4_word_search,
                   day_5_sum_mid_page)


def day_1(part='A') -> int:
    data = read_input(1)
    first = sorted(int(i.split()[0]) for i in data)
    second = sorted(int(i.split()[1]) for i in data)
    if part.upper() == 'A':
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
    ancestors, predecessors = defaultdict(set), defaultdict(set)
    for x, y in data:
        ancestors[x].add(y)
        predecessors[y].add(x)
    return day_5_sum_mid_page(ancestors, predecessors, updates, part)


if __name__ == '__main__':
    args = sys.argv[1:] if sys.argv[1:] else range(1, 26)
    args = [f'day_{i}' for i in args]
    members = inspect.getmembers(inspect.getmodule(inspect.currentframe()))
    funcs = {name: member for name, member in members
             if inspect.isfunction(member)}
    for day in args:
        if day not in funcs:
            print(f'{day}()= NotImplemented')
            continue
        print(f'{day}()= {funcs[day]()}')
        print(f'{day}(part="B")= {funcs[day](part="B")}')
