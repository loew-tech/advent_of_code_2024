import inspect
import sys


if __name__ == '__main__':
    args = sys.argv[1:]
    args = [f'day_{i}' for i in (args if args else range(1, 26))]
    members = inspect.getmembers(inspect.getmodule(inspect.currentframe()))
    funcs = {name: member for name, member in members
             if inspect.isfunction(member)}
    for day in args:
        if day not in funcs:
            print(f'{day}()=NotImplemented')
            continue
        print(f'{day}()={funcs[day]()}')
        print(f'{day}(part="B")={funcs[day](part="B")}')
