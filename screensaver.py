import os, sys, time, math, sqlite3, zlib, pickle
from dataclasses import dataclass
from collections import OrderedDict, deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor

IS_WINDOWS = (os.name == "nt")
CSI = "\x1b["

def ansi_home(): return CSI + "H"
def ansi_clear(): return CSI + "2J" + CSI + "H"
def ansi_hide_cursor(): return CSI + "?25l"
def ansi_show_cursor(): return CSI + "?25h"
def ansi_reset(): return CSI + "0m"
def bg_rgb(r,g,b): return f"{CSI}48;2;{r};{g};{b}m"

def enable_windows_vt() -> bool:
    if not IS_WINDOWS: return True
    try:
        import colorama
        colorama.just_fix_windows_console()
        return True
    except Exception:
        pass
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if k.GetConsoleMode(h, ctypes.byref(mode)) == 0: return False
        if k.SetConsoleMode(h, mode.value | 0x0004) == 0: return False
        return True
    except Exception:
        return False

# -------------------------
# Input (Windows msvcrt)
# -------------------------
if IS_WINDOWS:
    try:
        import msvcrt
        HAS_MSVCRT = True
    except Exception:
        HAS_MSVCRT = False
else:
    HAS_MSVCRT = False

def read_keys_msvcrt():
    keys = set()
    if not HAS_MSVCRT:
        return keys
    while msvcrt.kbhit():
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            ch2 = msvcrt.getwch()
            if ch2 == "H": keys.add("UP")
            if ch2 == "P": keys.add("DOWN")
            if ch2 == "K": keys.add("LEFT")
            if ch2 == "M": keys.add("RIGHT")
        else:
            keys.add(ch)
    return keys

# -------------------------
# Small deterministic helpers
# -------------------------
def u32(x): return np.asarray(x, dtype=np.uint32)

def hash_u32(a, b, c):
    a=u32(a); b=u32(b); c=u32(c)
    v = a * np.uint32(0x1F123BB5) ^ b * np.uint32(0x05491333) ^ c * np.uint32(0x2C1B3C6D)
    v ^= (v >> np.uint32(16))
    v *= np.uint32(0x7FEB352D)
    v ^= (v >> np.uint32(15))
    v *= np.uint32(0x846CA68B)
    v ^= (v >> np.uint32(16))
    return v

def rand01(v_u32):
    return ((v_u32 >> np.uint32(8)) & np.uint32(0x00FFFFFF)).astype(np.float32) / np.float32(0x01000000)

def fade(t): return t*t*(3.0-2.0*t)
def lerp(a,b,t): return a + (b-a)*t

def i64_to_u32(x): return (x.astype(np.int64) & np.int64(0xFFFFFFFF)).astype(np.uint32, copy=False)

def value_noise_2d(Xi, Yi, seed_u32: int, grain: float):
    g = max(1e-6, float(grain))
    xf = (Xi.astype(np.float64) / g)
    yf = (Yi.astype(np.float64) / g)
    x0 = np.floor(xf).astype(np.int64)
    y0 = np.floor(yf).astype(np.int64)
    tx = fade((xf - x0).astype(np.float32))
    ty = fade((yf - y0).astype(np.float32))

    x00=i64_to_u32(x0); y00=i64_to_u32(y0)
    x10=i64_to_u32(x0+1); y10=y00
    x01=x00; y01=i64_to_u32(y0+1)
    x11=x10; y11=y01

    s = np.uint32(seed_u32 & 0xFFFFFFFF)
    v00 = rand01(hash_u32(x00,y00,s))
    v10 = rand01(hash_u32(x10,y10,s))
    v01 = rand01(hash_u32(x01,y01,s))
    v11 = rand01(hash_u32(x11,y11,s))

    vx0 = lerp(v00,v10,tx)
    vx1 = lerp(v01,v11,tx)
    return lerp(vx0,vx1,ty).astype(np.float32, copy=False)

# -------------------------
# Colors + "hydra" palette generator
# -------------------------
ROOT_PALETTE = np.array([
    [ 36,  90, 178],   # water
    [ 56, 178,  76],   # prairie
    [140, 140, 140],   # mountain
], dtype=np.uint8)

BORDER_RGB = np.array([245,245,245], dtype=np.uint8)

def rgb_to_hsv(rgb01):
    r,g,b = rgb01[...,0], rgb01[...,1], rgb01[...,2]
    mx = np.maximum(np.maximum(r,g),b)
    mn = np.minimum(np.minimum(r,g),b)
    d = mx-mn
    h = np.zeros_like(mx)
    s = np.where(mx<=1e-12, 0.0, d/(mx+1e-12))
    v = mx
    m = d>1e-12
    idx = (mx==r)&m
    h[idx] = ((g[idx]-b[idx])/(d[idx]+1e-12))%6.0
    idx = (mx==g)&m
    h[idx] = ((b[idx]-r[idx])/(d[idx]+1e-12))+2.0
    idx = (mx==b)&m
    h[idx] = ((r[idx]-g[idx])/(d[idx]+1e-12))+4.0
    h = (h/6.0)%1.0
    return np.stack([h,s,v], axis=-1)

def hsv_to_rgb(hsv):
    h,s,v = hsv[...,0], hsv[...,1], hsv[...,2]
    i = (np.floor(h*6.0).astype(np.int32))%6
    f = (h*6.0)-np.floor(h*6.0)
    p = v*(1.0-s)
    q = v*(1.0-f*s)
    t = v*(1.0-(1.0-f)*s)
    r=np.zeros_like(v); g=np.zeros_like(v); b=np.zeros_like(v)
    for k in range(6):
        m = (i==k)
        if k==0: r[m],g[m],b[m]=v[m],t[m],p[m]
        if k==1: r[m],g[m],b[m]=q[m],v[m],p[m]
        if k==2: r[m],g[m],b[m]=p[m],v[m],t[m]
        if k==3: r[m],g[m],b[m]=p[m],q[m],v[m]
        if k==4: r[m],g[m],b[m]=t[m],p[m],v[m]
        if k==5: r[m],g[m],b[m]=v[m],p[m],q[m]
    return np.stack([r,g,b], axis=-1)

def child_palette_from_parent(parent_rgb_u8, salt_u32: int):
    """
    Returns 3 RGB uint8 colors, subtle variants of parent.
    Intended: all three are "related", and (roughly) average back to parent.
    """
    p = parent_rgb_u8.astype(np.float32)/255.0
    hsv = rgb_to_hsv(p)
    # deterministic small deltas
    h = int(hash_u32(np.uint32(salt_u32), np.uint32(parent_rgb_u8[0]), np.uint32(parent_rgb_u8[1])).item())
    dh = ((h & 0xFF)/255.0 - 0.5) * 0.06
    ds = (((h>>8)&0xFF)/255.0 - 0.5) * 0.08
    dv = (((h>>16)&0xFF)/255.0 - 0.5) * 0.08

    # symmetric variants around parent
    hsv0 = hsv.copy(); hsv1 = hsv.copy(); hsv2 = hsv.copy()
    hsv0[0] = (hsv0[0] - abs(dh)) % 1.0
    hsv2[0] = (hsv2[0] + abs(dh)) % 1.0
    hsv0[1] = np.clip(hsv0[1] - abs(ds)*0.5, 0, 1)
    hsv2[1] = np.clip(hsv2[1] + abs(ds)*0.5, 0, 1)
    hsv0[2] = np.clip(hsv0[2] - abs(dv)*0.5, 0, 1)
    hsv2[2] = np.clip(hsv2[2] + abs(dv)*0.5, 0, 1)
    # middle is close to parent, slightly sharpened
    hsv1[1] = np.clip(hsv1[1] + abs(ds)*0.25, 0, 1)
    hsv1[2] = np.clip(hsv1[2] + abs(dv)*0.25, 0, 1)

    rgb = np.stack([hsv_to_rgb(hsv0), hsv_to_rgb(hsv1), hsv_to_rgb(hsv2)], axis=0)
    return (np.clip(rgb*255.0, 0, 255)).astype(np.uint8)

def rgb_id(rgb_u8):
    return (int(rgb_u8[0])<<16) | (int(rgb_u8[1])<<8) | int(rgb_u8[2])

# -------------------------
# Connected components (4-neighbor) on integer IDs
# -------------------------
def cc4_int(grid_ids: np.ndarray):
    H,W = grid_ids.shape
    labels = -np.ones((H,W), dtype=np.int32)
    sizes = []
    q = deque()
    lab=0
    for y in range(H):
        for x in range(W):
            if labels[y,x] >= 0: continue
            t = grid_ids[y,x]
            labels[y,x] = lab
            q.append((y,x))
            sz=0
            while q:
                yy,xx = q.popleft(); sz += 1
                if yy>0 and labels[yy-1,xx]<0 and grid_ids[yy-1,xx]==t:
                    labels[yy-1,xx]=lab; q.append((yy-1,xx))
                if yy+1<H and labels[yy+1,xx]<0 and grid_ids[yy+1,xx]==t:
                    labels[yy+1,xx]=lab; q.append((yy+1,xx))
                if xx>0 and labels[yy,xx-1]<0 and grid_ids[yy,xx-1]==t:
                    labels[yy,xx-1]=lab; q.append((yy,xx-1))
                if xx+1<W and labels[yy,xx+1]<0 and grid_ids[yy,xx+1]==t:
                    labels[yy,xx+1]=lab; q.append((yy,xx+1))
            sizes.append(sz)
            lab += 1
    return labels, np.asarray(sizes, dtype=np.int32)

# -------------------------
# Disk cache for patches (sqlite)
# -------------------------
class PatchDiskCache:
    def __init__(self, path="scale_cache.sqlite"):
        self.path = path
        self.db = sqlite3.connect(self.path)
        self.db.execute("""
        CREATE TABLE IF NOT EXISTS patches(
          level INTEGER,
          x0 INTEGER, y0 INTEGER, w INTEGER, h INTEGER,
          key TEXT PRIMARY KEY,
          blob BLOB
        )
        """)
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_bbox ON patches(level, x0, y0, w, h)")
        self.db.commit()

    def put(self, level, x0,y0,w,h, key, obj):
        blob = zlib.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL), level=6)
        self.db.execute("INSERT OR REPLACE INTO patches(level,x0,y0,w,h,key,blob) VALUES(?,?,?,?,?,?,?)",
                        (int(level),int(x0),int(y0),int(w),int(h),str(key),sqlite3.Binary(blob)))
        self.db.commit()

    def query_overlapping(self, level, qx0,qy0,qx1,qy1, limit=512):
        # bbox overlap query
        cur = self.db.execute("""
        SELECT key, blob FROM patches
        WHERE level=?
          AND x0 <= ?
          AND y0 <= ?
          AND (x0 + w - 1) >= ?
          AND (y0 + h - 1) >= ?
        LIMIT ?
        """, (int(level), int(qx1), int(qy1), int(qx0), int(qy0), int(limit)))
        rows = cur.fetchall()
        out=[]
        for k, blob in rows:
            try:
                obj = pickle.loads(zlib.decompress(blob))
                out.append((k, obj))
            except Exception:
                continue
        return out

# -------------------------
# Patch in-memory + sampling
# -------------------------
@dataclass
class Patch:
    level: int          # child level this patch lives at
    x0: int
    y0: int
    w: int
    h: int
    parent_rgb: np.ndarray      # (3,) u8
    palette: np.ndarray         # (3,3) u8
    mask_bits: np.ndarray       # packed bits
    types: np.ndarray           # (h,w) u8 values 0..2

    def mask(self):
        m = np.unpackbits(self.mask_bits, bitorder='little')
        return m[:self.w*self.h].reshape(self.h,self.w).astype(bool)

class PatchLRU:
    def __init__(self, max_patches=256):
        self.max = int(max_patches)
        self.od = OrderedDict()  # key -> Patch

    def get(self, key):
        if key in self.od:
            self.od.move_to_end(key)
            return self.od[key]
        return None

    def put(self, key, patch: Patch):
        self.od[key] = patch
        self.od.move_to_end(key)
        while len(self.od) > self.max:
            self.od.popitem(last=False)

    def values(self):
        return list(self.od.values())

# -------------------------
# Sparse multiscale store with strict parent-deferral
# -------------------------
class ScaleStore:
    def __init__(self, seed=1337):
        self.seed = int(seed)
        self.disk = PatchDiskCache()
        self.lru = PatchLRU(max_patches=512)

        # outer invented containment overrides: level -> dict[(x,y)] = rgb_u8
        self.outer_override = {}
        # per-level cell overrides (level -> {(x,y): rgb_u8}) used for invented parent cells
        self.cell_override = {}

    def set_cell_override(self, level: int, x: int, y: int, rgb_u8: np.ndarray):
        d = self.cell_override.setdefault(int(level), {})
        d[(int(x), int(y))] = rgb_u8.copy()

    def get_cell_override(self, level: int, x: int, y: int):
        d = self.cell_override.get(int(level))
        if d is None:
            return None
        v = d.get((int(x), int(y)))
        return None if v is None else v.copy()

    def _root_cell_rgb(self, x0, y0):
        # Root is the only globally-defined field
        s = np.uint32(self.seed & 0xFFFFFFFF)
        X = np.array([[x0]], dtype=np.int64)
        Y = np.array([[y0]], dtype=np.int64)
        n0 = value_noise_2d(X, Y, int(s ^ 0xA531), 11.0)[0,0]
        n1 = value_noise_2d(X, Y, int(s ^ 0x19D3),  7.0)[0,0]
        n2 = value_noise_2d(X, Y, int(s ^ 0xD1B5),  4.0)[0,0]
        t = int(np.argmax([n0,n1,n2]))
        return ROOT_PALETTE[t].copy()

    def _outer_bg_rgb(self, level, x, y):
        # deterministic "outside" for negative levels, but only visible after containment exists
        s = np.uint32((self.seed ^ (level*2654435761)) & 0xFFFFFFFF)
        X = np.array([[x]], dtype=np.int64)
        Y = np.array([[y]], dtype=np.int64)
        a = value_noise_2d(X,Y,int(s^0x55AA), 9.0)[0,0]
        b = value_noise_2d(X,Y,int(s^0xAA55), 5.0)[0,0]
        t = int(np.argmax([a,b,0.5*(a+b)]))
        return ROOT_PALETTE[t].copy()

    def _load_patches_overlapping(self, level, x0,y0,x1,y1):
        rows = self.disk.query_overlapping(level, x0,y0,x1,y1, limit=2048)
        for key, obj in rows:
            if self.lru.get(key) is not None:
                continue
            try:
                patch = Patch(**obj)
                self.lru.put(key, patch)
            except Exception:
                pass

    def prefetch(self, level, x0,y0,x1,y1):
        self._load_patches_overlapping(level, x0,y0,x1,y1)

    def sample_rgb(self, level: int, x: int, y: int):
        """
        CRITICAL INVARIANT:
        - If there is no instantiated patch at `level` covering (x,y),
          this returns the parent sample at (level-1, x>>1, y>>1).
        - Therefore zooming does not reveal new detail.
        """
        level = int(level); x=int(x); y=int(y)

        # per-level cell override (e.g. invented parent cells)
        ov = self.get_cell_override(level, x, y)
        if ov is not None:
            return ov

        # Outer invented containment cell overrides
        if level < 0:
            ov = self.outer_override.get(level)
            if ov is not None:
                # If inside an invented cell (we use exact match on a coarse grid)
                if (x,y) in ov:
                    return ov[(x,y)]
            # background only meaningful once we've wrapped out; ok for prototype
            return self._outer_bg_rgb(level, x, y)

        # check patches at this level
        for patch in self.lru.values():
            if patch.level != level:
                continue
            if x < patch.x0 or y < patch.y0 or x >= patch.x0+patch.w or y >= patch.y0+patch.h:
                continue
            lx = x - patch.x0
            ly = y - patch.y0
            m = patch.mask()
            if not m[ly,lx]:
                break
            t = int(patch.types[ly,lx]) % 3
            return patch.palette[t].copy()

        # No instantiated content at this level => defer to parent (or root)
        if level == 0:
            return self._root_cell_rgb(x, y)
        return self.sample_rgb(level-1, x>>1, y>>1)

    def invent_outer_containing_cell(self, new_level: int, contain_x: int, contain_y: int, rgb_u8: np.ndarray):
        """
        This is your "draw a boundary, make a cell that contains the view, then zoom out into environment".
        """
        new_level = int(new_level)
        d = self.outer_override.setdefault(new_level, {})
        d[(int(contain_x), int(contain_y))] = rgb_u8.copy()

    def write_patch(self, patch: Patch, key: str):
        obj = dict(
            level=int(patch.level),
            x0=int(patch.x0), y0=int(patch.y0), w=int(patch.w), h=int(patch.h),
            parent_rgb=patch.parent_rgb, palette=patch.palette,
            mask_bits=patch.mask_bits, types=patch.types
        )
        self.disk.put(patch.level, patch.x0, patch.y0, patch.w, patch.h, key, obj)
        self.lru.put(key, patch)


def build_child_patch(child_level, cx0, cy0, cw, ch, parent_rgb, palette, cmask, seed, region_salt):
    Xc = (cx0 + np.arange(cw, dtype=np.int64))[None,:].repeat(ch, axis=0)
    Yc = (cy0 + np.arange(ch, dtype=np.int64))[:,None].repeat(cw, axis=1)

    s = (seed ^ (child_level*2654435761) ^ region_salt) & 0xFFFFFFFF
    n0 = value_noise_2d(Xc, Yc, int(s ^ 0xA531), 10.0)
    n1 = value_noise_2d(Xc, Yc, int(s ^ 0x19D3),  6.0)
    n2 = value_noise_2d(Xc, Yc, int(s ^ 0xD1B5),  4.0)
    logits = np.stack([n0,n1,n2], axis=0)

    cxn = cx0 + cw//2
    cyn = cy0 + ch//2
    dx = (Xc.astype(np.float32) - float(cxn))
    dy = (Yc.astype(np.float32) - float(cyn))
    bump = np.exp(-(dx*dx+dy*dy)/(2.0*(8.0**2))).astype(np.float32)
    logits[1] += 0.35*bump
    logits[2] += 0.20*bump

    types = np.argmax(logits, axis=0).astype(np.uint8)
    types[~cmask] = np.uint8(1)

    for _ in range(2):
        t = types
        up = np.roll(t, -1, axis=0)
        dn = np.roll(t,  1, axis=0)
        lf = np.roll(t, -1, axis=1)
        rt = np.roll(t,  1, axis=1)
        v0 = (t==0).astype(np.int16)+(up==0)+(dn==0)+(lf==0)+(rt==0)
        v1 = (t==1).astype(np.int16)+(up==1)+(dn==1)+(lf==1)+(rt==1)
        v2 = (t==2).astype(np.int16)+(up==2)+(dn==2)+(lf==2)+(rt==2)
        newt = np.argmax(np.stack([v0,v1,v2], axis=0), axis=0).astype(np.uint8)
        types = np.where(cmask, newt, types)

    mask_bits = np.packbits(cmask.reshape(-1).astype(np.uint8), bitorder='little')
    return Patch(
        level=int(child_level),
        x0=int(cx0), y0=int(cy0), w=int(cw), h=int(ch),
        parent_rgb=parent_rgb.copy(),
        palette=palette.copy(),
        mask_bits=mask_bits.copy(),
        types=types.copy()
    )

# -------------------------
# View + engine logic (atomic transitions only)
# -------------------------
@dataclass
class View:
    cx: float = 0.0   # world center in root-cell units
    cy: float = 0.0
    level: int = 0
    log2_wupp: float = -5.0   # world units per pixel

    def wupp(self): return 2.0 ** self.log2_wupp
    def scale(self): return 2.0 ** self.level  # cells per world unit at current level

class Engine:
    def __init__(self, H=24, W=48, seed=1337):
        self.H, self.W = int(H), int(W)
        self.store = ScaleStore(seed=seed)
        self.view = View()
        self.hydra_pulse = 0.0

        # Parameters you can tune without changing semantics:
        self.zoom_speed = 1.25
        self.pan_speed = 1.0
        self.refine_tol = 1.00  # refine when view_radius <= mean_radius * refine_tol
        self.max_gate_grid = (140, 100)  # cap grid used for CC gate

        self.last_event = ""
        self.last_gate = {"ok": False, "mean_r": float("inf"), "view_r": float("inf"), "cells": (0,0), "grid": (0,0)}

        # async patch generation
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = []  # list of (future, key)
        # small cache to avoid recomputing the gate too often
        self._gate_cache = {"t": 0.0, "log2": None, "g": None}


    # ----- attention bbox in current level cell coords -----
    def attention_bbox(self, margin_cells=8):
        v = self.view
        wupp = v.wupp()
        fov_w = self.W * wupp
        fov_h = self.H * wupp
        x0w = v.cx - 0.5*fov_w
        x1w = v.cx + 0.5*fov_w
        y0w = v.cy - 0.5*fov_h
        y1w = v.cy + 0.5*fov_h
        s = v.scale()
        x0 = int(math.floor(x0w*s)) - margin_cells
        x1 = int(math.floor(x1w*s)) + margin_cells
        y0 = int(math.floor(y0w*s)) - margin_cells
        y1 = int(math.floor(y1w*s)) + margin_cells
        return x0,y0,x1,y1

    # ----- gate computation on a capped grid -----
    def compute_gate(self):
        # cheap temporal cache to avoid recomputing gate every frame
        t = time.time()
        if (self._gate_cache["g"] is not None and
            self._gate_cache["log2"] is not None and
            abs(self.view.log2_wupp - self._gate_cache["log2"]) < 0.08 and
            (t - self._gate_cache["t"]) < 0.12):
            self.last_gate = self._gate_cache["g"]
            return self.last_gate

        v = self.view
        s = v.scale()
        wupp = v.wupp()
        fov_w = self.W * wupp
        fov_h = self.H * wupp
        cw = max(1, int(math.ceil(fov_w * s)))
        ch = max(1, int(math.ceil(fov_h * s)))

        # view "radius" as circle from area
        view_area = float(cw*ch)
        view_r = math.sqrt(view_area / math.pi)

        # build a grid for CC, but cap for speed
        maxw, maxh = self.max_gate_grid
        gw = min(cw, maxw)
        gh = min(ch, maxh)

        # attention grid covers the view, centered; sample step skips if huge
        step_x = max(1, cw // gw)
        step_y = max(1, ch // gh)

        x0,y0,x1,y1 = self.attention_bbox(margin_cells=0)
        xs = np.arange(x0, x0 + gw*step_x, step_x, dtype=np.int64)
        ys = np.arange(y0, y0 + gh*step_y, step_y, dtype=np.int64)

        X,Y = np.meshgrid(xs, ys)
        # prefetch patches for current level bbox
        self.store.prefetch(v.level, int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

        # sample RGB IDs grid (region equality is by exact color)
        ids = np.empty((gh,gw), dtype=np.int32)
        for j in range(gh):
            for i in range(gw):
                rgb = self.store.sample_rgb(v.level, int(X[j,i]), int(Y[j,i]))
                ids[j,i] = rgb_id(rgb)

        labels, sizes = cc4_int(ids)
        mean_area = float(sizes.mean()) * float(step_x*step_y) if sizes.size else 1.0
        mean_r = math.sqrt(mean_area / math.pi)

        ok = (view_r <= mean_r * self.refine_tol)

        self.last_gate = {"ok": ok, "mean_r": mean_r, "view_r": view_r, "cells": (cw,ch), "grid": (gw,gh),
                  "x0": x0, "y0": y0, "step": (step_x, step_y), "ids": ids, "labels": labels}
        # update cache
        self._gate_cache = {"t": t, "log2": self.view.log2_wupp, "g": self.last_gate}
        return self.last_gate

    def poll_jobs(self):
        if not self.pending:
            return
        still = []
        for fut, key in self.pending:
            if fut.done():
                try:
                    patch = fut.result()
                    self.store.write_patch(patch, key)
                    self.last_event = f"PATCH READY {key}"
                except Exception as e:
                    self.last_event = f"PATCH FAIL {e}"
            else:
                still.append((fut, key))
        self.pending = still

    # ----- masked child generation for center region(s) -----
    def refine_atomic_if_gate(self, force=False):
        g = self.compute_gate()
        if (not force) and (not g["ok"]):
            return

        v = self.view
        L = v.level
        # define a small region-grid around center using the gate grid
        ids = g["ids"]
        labels = g["labels"]
        gh, gw = ids.shape
        cy = gh//2
        cx = gw//2
        center_lab = int(labels[cy,cx])

        # include any region labels that touch the center cell (share boundary)
        labs = {center_lab}
        for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            yy = cy+dy; xx = cx+dx
            if 0 <= yy < gh and 0 <= xx < gw:
                labs.add(int(labels[yy,xx]))

        region_mask = np.isin(labels, np.array(list(labs), dtype=np.int32))
        if not np.any(region_mask):
            return

        # compute bbox in CURRENT LEVEL cell coords for that region_mask
        x0 = g["x0"]; y0 = g["y0"]
        step_x, step_y = g["step"]
        ys, xs = np.where(region_mask)
        bx0 = x0 + int(xs.min())*step_x
        by0 = y0 + int(ys.min())*step_y
        bx1 = x0 + int(xs.max())*step_x + (step_x-1)
        by1 = y0 + int(ys.max())*step_y + (step_y-1)

        # parent color at the exact view center cell
        center_cell_x = int(math.floor(v.cx * v.scale()))
        center_cell_y = int(math.floor(v.cy * v.scale()))
        parent_rgb = self.store.sample_rgb(L, center_cell_x, center_cell_y)

        # child level and child bbox
        child_level = L + 1
        cx0 = 2*bx0; cy0 = 2*by0
        cw = 2*(bx1-bx0+1); ch = 2*(by1-by0+1)

        # build a high-res mask by upsampling the region mask (nearest)
        # but also expand it to cover full step_x/step_y tiles
        pmask = np.zeros((by1-by0+1, bx1-bx0+1), dtype=bool)
        for j,i in zip(ys,xs):
            # fill the block corresponding to this sampled cell
            px0 = (x0 + i*step_x) - bx0
            py0 = (y0 + j*step_y) - by0
            pmask[py0:py0+step_y, px0:px0+step_x] = True
        cmask = np.repeat(np.repeat(pmask, 2, axis=0), 2, axis=1)

        # generate child palette (3 related colors)
        region_salt = int(hash_u32(np.uint32(child_level), np.uint32(bx0 & 0xFFFFFFFF), np.uint32(by0 & 0xFFFFFFFF)).item())
        palette = child_palette_from_parent(parent_rgb, region_salt)

        # schedule patch generation off-thread; keep using parent-deferred sampling until ready
        key = f"L{child_level}:{cx0}:{cy0}:{cw}:{ch}:{region_salt}"
        fut = self.executor.submit(
            build_child_patch,
            child_level, cx0, cy0, cw, ch,
            parent_rgb, palette, cmask,
            self.store.seed, region_salt
        )
        self.pending.append((fut, key))

        # ATOMIC swap immediately (child content defers to parent until patch arrives)
        v.level = child_level
        v.log2_wupp -= 1.0
        self.hydra_pulse = 1.0
        self.last_event = f"REFINE scheduled L={v.level} key={key}"

    def coarsen_atomic(self):
        v = self.view
        new_level = v.level - 1

        new_scale = 2.0 ** new_level
        contain_x = int(math.floor(v.cx * new_scale))
        contain_y = int(math.floor(v.cy * new_scale))

        # Average color of "now" under current view
        wupp = v.wupp()
        fov_w = self.W * wupp
        fov_h = self.H * wupp
        s = 2.0 ** v.level
        cw = max(8, int(min(64, math.ceil(fov_w * s))))
        ch = max(6, int(min(48, math.ceil(fov_h * s))))
        xs = np.linspace(v.cx - 0.5*fov_w, v.cx + 0.5*fov_w, cw, endpoint=False)
        ys = np.linspace(v.cy - 0.5*fov_h, v.cy + 0.5*fov_h, ch, endpoint=False)

        acc = np.zeros(3, dtype=np.float64)
        for yy in ys:
            for xx in xs:
                cx = int(math.floor(xx * s))
                cy = int(math.floor(yy * s))
                rgb = self.store.sample_rgb(v.level, cx, cy)
                acc += rgb.astype(np.float64)
        avg = np.clip(acc / float(cw*ch), 0, 255).astype(np.uint8)

        # Invent the containing parent cell at new_level (ALWAYS)
        if new_level < 0:
            self.store.invent_outer_containing_cell(new_level, contain_x, contain_y, avg)
        else:
            self.store.set_cell_override(new_level, contain_x, contain_y, avg)

        # Visible boundary ring (4-neighbor ring = "geometry says 4")
        ring = [(contain_x-1, contain_y), (contain_x+1, contain_y),
                (contain_x, contain_y-1), (contain_x, contain_y+1)]
        for rx, ry in ring:
            if new_level < 0:
                self.store.invent_outer_containing_cell(new_level, rx, ry, BORDER_RGB)
            else:
                self.store.set_cell_override(new_level, rx, ry, BORDER_RGB)

        # ATOMIC swap outward
        v.level = new_level
        v.log2_wupp += 1.0
        self.hydra_pulse = 1.0
        self.last_event = f"COARSEN to L={v.level} contain=({contain_x},{contain_y})"

    def step(self, dt: float, zoom_dir: int):
        # decay hydra pulse (pure visual)
        self.hydra_pulse *= math.exp(-2.2*dt)

        # pure zoom transform
        if zoom_dir != 0:
            self.view.log2_wupp += (self.zoom_speed*dt) * (1.0 if zoom_dir>0 else -1.0)
        self.view.log2_wupp = float(np.clip(self.view.log2_wupp, -26.0, +12.0))

    def render(self, allow_color=True):
        v = self.view
        wupp = v.wupp()
        fov_w = self.W*wupp
        fov_h = self.H*wupp
        x_min = v.cx - 0.5*fov_w
        y_min = v.cy - 0.5*fov_h
        s = v.scale()

        # prefetch patches for now, parent, child in attention bbox
        x0,y0,x1,y1 = self.attention_bbox(margin_cells=12)
        self.store.prefetch(v.level, x0,y0,x1,y1)
        self.store.prefetch(v.level-1, x0>>1,y0>>1,x1>>1,y1>>1)
        self.store.prefetch(v.level+1, x0<<1,y0<<1,x1<<1,y1<<1)

        lines = []
        for j in range(self.H):
            parts = []
            wy = y_min + (j+0.5)*(fov_h/self.H)
            for i in range(self.W):
                wx = x_min + (i+0.5)*(fov_w/self.W)
                cx = int(math.floor(wx*s))
                cy = int(math.floor(wy*s))
                rgb = self.store.sample_rgb(v.level, cx, cy)

                # hydra pulse: blend slightly toward palette variants if this pixel is in a patch
                # (we keep it subtle and purely visual; it does not change the stored map)
                p = float(np.clip(self.hydra_pulse, 0.0, 1.0))
                if p > 1e-3:
                    # attempt to find a patch covering this cell to determine its parent palette
                    # (cheap scan of LRU)
                    for patch in self.store.lru.values():
                        if patch.level != v.level: continue
                        if cx < patch.x0 or cy < patch.y0 or cx >= patch.x0+patch.w or cy >= patch.y0+patch.h:
                            continue
                        m = patch.mask()
                        lx,ly = cx-patch.x0, cy-patch.y0
                        if not m[ly,lx]:
                            break
                        t = int(patch.types[ly,lx])%3
                        hyd = patch.palette[t].astype(np.float32)
                        rgb = (rgb.astype(np.float32)*(1.0-p) + hyd*p).astype(np.uint8)
                        break

                if not allow_color:
                    # ascii fallback
                    if rgb_id(rgb) == rgb_id(ROOT_PALETTE[0]): ch = "~"
                    elif rgb_id(rgb) == rgb_id(ROOT_PALETTE[1]): ch = "."
                    else: ch = "^"
                    parts.append(ch)
                else:
                    parts.append(bg_rgb(int(rgb[0]), int(rgb[1]), int(rgb[2])) + "  ")
            if allow_color:
                parts.append(ansi_reset())
            lines.append("".join(parts))
        return lines

# -------------------------
# Main (Windows ANSI loop)
# -------------------------
def main():
    H,W = 24,48
    eng = Engine(H=H, W=W, seed=1337)
    vt_ok = enable_windows_vt()

    sys.stdout.write(ansi_hide_cursor())
    sys.stdout.write(ansi_clear())
    sys.stdout.flush()

    last = time.time()
    try:
        while True:
            now = time.time()
            dt = max(1e-4, now-last)
            last = now

            keys = read_keys_msvcrt() if IS_WINDOWS else set()
            if ("x" in keys) or ("X" in keys):
                break

            # pan: pure view translation
            pan = eng.pan_speed * eng.view.wupp() * dt
            if ("w" in keys) or ("UP" in keys): eng.view.cy -= pan
            if ("s" in keys) or ("DOWN" in keys): eng.view.cy += pan
            if ("a" in keys) or ("LEFT" in keys): eng.view.cx -= pan
            if ("d" in keys) or ("RIGHT" in keys): eng.view.cx += pan

            zoom_dir = 0
            if ("e" in keys) or ("+" in keys) or ("=" in keys): zoom_dir = -1
            if ("q" in keys) or ("-" in keys) or ("_" in keys): zoom_dir = +1

            eng.step(dt, zoom_dir)

            # Atomic transitions:
            # - refine happens only when zooming in AND the mean-radius gate passes (or forced)
            if "]" in keys:
                eng.refine_atomic_if_gate(force=True)
            elif "[" in keys:
                eng.coarsen_atomic()
            else:
                if zoom_dir < 0:
                    eng.refine_atomic_if_gate(force=False)
                # zoom-out coarsen is explicit via '[' in this prototype

            g = eng.last_gate
            header = [
                f"L={eng.view.level:>3d}  log2_wupp={eng.view.log2_wupp:+.3f}  wupp={eng.view.wupp():.3g}  hydra={eng.hydra_pulse:.3f}",
                f"gate: ok={g['ok']}  view_r={g['view_r']:.2f}  mean_r={g['mean_r']:.2f}  view_cells={g['cells']}  grid={g['grid']}",
                f"patches_in_ram={len(eng.store.lru.od)}  event: {eng.last_event}",
                "WASD/arrows pan | hold E/+ zoom in | hold Q/- zoom out | ] force refine | [ coarsen | X quit",
                ""
            ]

            lines = eng.render(allow_color=vt_ok)

            sys.stdout.write(ansi_home())
            sys.stdout.write("\n".join(header))
            sys.stdout.write("\n".join(lines))
            sys.stdout.write(ansi_reset())
            sys.stdout.flush()
            time.sleep(0.016)

    finally:
        sys.stdout.write(ansi_reset())
        sys.stdout.write(ansi_show_cursor())
        sys.stdout.flush()

if __name__ == "__main__":
    main()
