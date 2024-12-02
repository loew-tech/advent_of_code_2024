from collections import Counter
import inspect
from operator import lt, gt
import sys

from utils import read_input


def day_1(part='A') -> int:
    data = read_input(1)
    first = sorted(int(i.split()[0]) for i in data)
    second = sorted(int(i.split()[1]) for i in data)
    if part.upper() == 'A':
        return sum(abs(v-second[i]) for i, v in enumerate(first))
    counter = Counter(second)
    return sum(v*counter.get(v, 0) for v in first)


def day_2(part='A') -> int:
    data = [[int(i) for i in report.split()] for report in read_input(2)]
    valid_count = 0
    for report in data:
        if len(report) == 1:
            valid_count += 1
            continue
        valid, op_ = True, gt if report[1] > report[0] else lt
        for i, v in enumerate(report[1:]):
            valid = valid and op_(v, report[i]) and abs(report[i]-v) <= 3
        valid_count += valid
    return valid_count


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
