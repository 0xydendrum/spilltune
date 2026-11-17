"""Hardware probe: GPU memory (nvidia-smi) and system RAM, without third-party packages."""
import ctypes
import os
import subprocess
import sys
from dataclasses import dataclass, field

MiB = 2 ** 20


@dataclass
class GPU:
    index: int
    name: str
    total_mib: int
    used_mib: int

    @property
    def free_mib(self):
        return self.total_mib - self.used_mib


@dataclass
class Hardware:
    gpus: list = field(default_factory=list)
    ram_total_mib: int = 0
    ram_free_mib: int = 0

    @property
    def vram_total_mib(self):
        return sum(g.total_mib for g in self.gpus)

    def describe(self):
        g = ", ".join(f"{x.name} {x.total_mib / 1024:.0f} GB ({x.used_mib / 1024:.1f} GB in use)" for x in self.gpus)
        return f"{g or 'no NVIDIA GPU found'}; RAM {self.ram_total_mib / 1024:.0f} GB ({self.ram_free_mib / 1024:.0f} GB free)"


def gpus():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    res = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4 and parts[0].isdigit():
            res.append(GPU(int(parts[0]), parts[1], int(float(parts[2])), int(float(parts[3]))))
    return res


def ram():
    """(total MiB, available MiB)."""
    if sys.platform == "win32":
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys // MiB, m.ullAvailPhys // MiB
    if os.path.exists("/proc/meminfo"):
        info = {}
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            info[k] = int(v.split()[0]) // 1024
        return info.get("MemTotal", 0), info.get("MemAvailable", info.get("MemFree", 0))
    try:
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) // MiB
        return total, total // 2
    except (OSError, ValueError):
        return 0, 0


def probe():
    t, f = ram()
    return Hardware(gpus=gpus(), ram_total_mib=t, ram_free_mib=f)
