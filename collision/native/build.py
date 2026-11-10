"""Explicit build; production never compiles or switches backends implicitly."""
import json,subprocess,sys,time
from pathlib import Path

root=Path(__file__).resolve().parent
command=[sys.executable,'-m','ziglang','c++','-std=c++17','-O3','-shared','-fPIC','-fno-exceptions','-fno-rtti',str(root/'kernel.cpp'),'-o',str(root/'libgaussian.so')]
t=time.perf_counter();subprocess.run(command,check=True)
result={'command':command,'compile_wall_seconds':time.perf_counter()-t}
print(json.dumps(result,indent=2))
