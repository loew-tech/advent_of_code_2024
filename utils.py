from datetime import datetime
from http import HTTPStatus
import requests
from typing import List

ADVENT_URI = 'https://adventofcode.com/'


def read_input(day: int | str, delim='\n', year=None) -> List[str]:
    year = year if year is not None else datetime.now().year
    with open('.env') as env_:
        session_id = env_.read()
    response = requests.get(f'{ADVENT_URI}{year}/day/{day}/input',
                            cookies={'session': session_id})
    if response.status_code == HTTPStatus.OK:
        return response.text.split(delim)[:-1] if delim else response.text


def day_2_helper(part='A') -> int:
    count = 0
    for report in (r.split() for r in read_input(2)):
        if part.upper() == 'A':
            count += _is_valid_report(report)
        else:
            for i in range(len(report)):
                if _is_valid_report(report[:i]+report[i+1:]):
                    count += 1
                    break
    return count


def _is_valid_report(report: List[str]) -> int:
    diffs = {int(v)-int(report[i]) for i, v in enumerate(report[1:])}
    return diffs <= {1, 2, 3} or diffs <= {-1, -2, -3}

