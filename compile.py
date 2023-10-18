#!/usr/bin/env python3
# quick and dirty utility script
import os

opt = input("""1) Native (Debug)
2) Web
3) start local HTTP server
> """)
if opt == "1":
    os.system("cargo run")
elif opt == "2":
    os.system("wasm-pack build --target web")
else:
    os.system("python3 -m http.server")