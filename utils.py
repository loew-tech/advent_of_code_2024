from collections import defaultdict
from datetime import datetime
from http import HTTPStatus
import re
import requests
from typing import List, Callable, Set

from constants import DIRECTIONS

ADVENT_URI = 'https://adventofcode.com/'


def read_input(day: int | str, delim='\n', year=None) -> List[str]:
    year = year if year is not None else datetime.now().year
    with open('.env') as env_:
        session_id = env_.read()
    response = requests.get(f'{ADVENT_URI}{year}/day/{day}/input',
                            cookies={'session': session_id})
    if response.status_code == HTTPStatus.OK:
        return response.text.split(delim)[:-1] if delim else response.text


def get_inbounds(grid: List[List[any] | str]) -> Callable[[int, int], bool]:
    return lambda y, x: inbounds(y, x, grid)


def inbounds(y, x: int, grid: List[List[any] | str]) -> bool:
    return 0 <= y < len(grid) and 0 <= x < len(grid[y])


def day_2_helper(part='A') -> int:
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


def _is_valid_report(report: List[str]) -> int:
    diffs = {int(v) - int(report[i]) for i, v in enumerate(report[1:])}
    return diffs <= {1, 2, 3} or diffs <= {-1, -2, -3}


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


def day_5_sum_mid_page(ancestors, predecessors: defaultdict[Set[int]],
                       updates: List[List[int]], part='A') -> int:
    def is_ordered(update_: List[int]) -> bool:
        observed = set()
        for page in update_:
            if observed & ancestors[page]:
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
            sum_ += (part.upper() == 'A') * update[len(update)//2]
            continue
        if part.upper() == 'A':
            continue
        fixed = fix_update(update)
        sum_ += fixed[len(fixed)//2]
    return sum_
