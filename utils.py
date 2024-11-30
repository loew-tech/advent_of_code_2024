import datetime
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
        return response.text.split(delim)
