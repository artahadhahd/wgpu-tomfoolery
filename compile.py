#!/usr/bin/env python3
import os
import sys

opts = \
"""
1) Run [Debug]
2) WASM
3) Start local HTTP server
4) Run [Release]
5) Build [Debug]
6) Build [Release]
"""

opt_to_func: dict[str, str] = {
    '1' : 'cargo run',
    '2' : 'wasm-pack build --target web',
    '3' : 'python3 -m http.server',
    '4' : 'cargo run --release',
    '5' : 'cargo build',
    '6' : 'cargo build --release',
}

print(opts)
def main() -> int:
    opt: str
    try:
        opt = input("> ")
    except EOFError:
        print()
        return 1
    if opt not in opt_to_func.keys():
        print("Invalid option")
        return main()
    os.system(opt_to_func[opt])
    return 0

if __name__ == '__main__':
    sys.exit(main())