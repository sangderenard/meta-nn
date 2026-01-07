import math
import random
from dataclasses import dataclass

import numpy as np
import pygame


class Species:
    def __init__(
        self,
        name,
        hue_deg,
        decay,
        diffusion=None,
        metric_enabled=False,
        metric_decay=0.0,
        metric_diffusion=0.0,
    ):
        self.name = name
        self.hue_deg = float(hue_deg)
        self.decay = float(decay)
        self.diffusion = diffusion
        self.metric_enabled = bool(metric_enabled)
        self.metric_decay = float(metric_decay)
        self.metric_diffusion = float(metric_diffusion)

    def step_concentration(self, grid_slice, dt, diffusion):
        diff = self.diffusion if self.diffusion is not None else diffusion
        out = diffuse_grid(grid_slice, diff, dt)
        if self.decay > 0.0:
            out *= max(0.0, 1.0 - self.decay * dt)
        return out

    def step_metric(self, metric_slice, dt):
        if not self.metric_enabled:
            return metric_slice
        out = diffuse_grid(metric_slice, self.metric_diffusion, dt)
        if self.metric_decay > 0.0:
            out *= max(0.0, 1.0 - self.metric_decay * dt)
        return out


@dataclass
class GridSource:
    x: int
    y: int
    species: int
    rate: float


@dataclass
class NodeSource:
    node: int
    species: int
    rate: float


@dataclass
class SubgraphRule:
    name: str
    applicable_kinds: tuple  # (start_kind, end_kind) or ("any","any")
    new_nodes: list          # list of (u, v) in edge-local coords, u along edge, v normal
    edges: list              # list of (a, b) using indices: 0=start, 1=end, 2+=new nodes
    new_kinds: list          # kind labels for new nodes


class Filter:
    def apply(self, conc, dt):
        return conc


class DecayFilter(Filter):
    def __init__(self, rate):
        self.rate = rate

    def apply(self, conc, dt):
        return conc * max(0.0, 1.0 - self.rate * dt)


class ConvertFilter(Filter):
    def __init__(self, src, dst, rate):
        self.src = src
        self.dst = dst
        self.rate = rate

    def apply(self, conc, dt):
        amount = conc[self.src] * self.rate * dt
        conc[self.src] = max(0.0, conc[self.src] - amount)
        conc[self.dst] = conc[self.dst] + amount
        return conc


class ClampFilter(Filter):
    def __init__(self, lo=0.0, hi=1.0):
        self.lo = lo
        self.hi = hi

    def apply(self, conc, dt):
        return np.clip(conc, self.lo, self.hi)


class VoxelGrid:
    def __init__(self, width, height, species_count):
        self.width = width
        self.height = height
        self.species_count = species_count
        self.grid = np.zeros((species_count, height, width), dtype=np.float32)
        self.metric = np.zeros((species_count, 2, 2, height, width), dtype=np.float32)
        self.sources = []

    def add_source(self, x, y, species, rate):
        self.sources.append(GridSource(x, y, species, rate))

    def add_blob(self, cx, cy, species, amount, radius=2):
        for y in range(max(0, cy - radius), min(self.height, cy + radius + 1)):
            for x in range(max(0, cx - radius), min(self.width, cx + radius + 1)):
                dx = x - cx
                dy = y - cy
                if dx * dx + dy * dy <= radius * radius:
                    self.grid[species, y, x] += amount

    def step(self, dt, diffusion, species_defs):
        for src in self.sources:
            if 0 <= src.x < self.width and 0 <= src.y < self.height:
                self.grid[src.species, src.y, src.x] += src.rate * dt
        for s in range(self.species_count):
            self.grid[s] = species_defs[s].step_concentration(self.grid[s], dt, diffusion)
            if species_defs[s].metric_enabled:
                for i in range(2):
                    for j in range(2):
                        self.metric[s, i, j] = species_defs[s].step_metric(
                            self.metric[s, i, j], dt
                        )
                low = self.grid[s] <= 1e-6
                if np.any(low):
                    self.metric[s, :, :, low] = 0.0
        self.grid = np.clip(self.grid, 0.0, 1.0)

    def sample_metric_mean(self, species_idx, x, y, eps=1e-6):
        if species_idx < 0 or species_idx >= self.species_count:
            return np.zeros((2, 2), dtype=np.float32)
        conc = float(self.grid[species_idx, y, x])
        if conc <= eps:
            return np.zeros((2, 2), dtype=np.float32)
        return self.metric[species_idx, :, :, y, x] / conc


class NodeType:
    name = "stem"
    impedance_species = None
    emit = None
    consume = None
    motility_species = None
    motility_gain = 0.0
    drive_weights = None
    stiffness = 0.3

    def on_pre_step(self, world, node_idx, dt):
        pass

    def on_post_step(self, world, node_idx, dt):
        pass

    def on_exchange(self, world, node_idx, dt):
        pass

    def on_speciate(self, env, current_kind):
        return current_kind


class WaterRoot(NodeType):
    name = "water_root"
    impedance_species = 2
    emit = {"agar": 0.01}
    consume = {"moisture": 0.02}
    stiffness = 0.35


class AirRoot(NodeType):
    name = "air_root"
    impedance_species = 1
    emit = {"agar": 0.008}
    consume = {"nutrient": 0.015}
    stiffness = 0.3


class Stem(NodeType):
    name = "stem"
    impedance_species = 0
    emit = {"agar": 0.015}
    stiffness = 0.7


class Leaf(NodeType):
    name = "leaf"
    impedance_species = 3
    emit = {"agar": 0.025}
    stiffness = 0.45


class Slime(NodeType):
    name = "slime"
    impedance_species = 4
    emit = {"signal": 0.06}
    consume = {"agar": 0.04}
    motility_species = "agar"
    motility_gain = 1.2
    drive_weights = {"explore": 1.1, "exploit": 1.4, "wander": 0.35}
    stiffness = 0.08
    metabolism_rate = 0.06

    def on_pre_step(self, world, node_idx, dt):
        if world.agar_index < 0:
            return
        agar = world.graph.node_conc[node_idx, world.agar_index]
        burn = min(agar, self.metabolism_rate * dt)
        world.graph.node_conc[node_idx, world.agar_index] = agar - burn


def step_node_physics(
    graph,
    grid,
    dt,
    bounds,
    damping=0.2,
    density_imped=1.5,
    radius=0.32,
    impedance_species=None,
):
    if graph.node_pos.size == 0:
        return
    w, h = bounds
    ix = np.clip(np.round(graph.node_pos[:, 0]).astype(int), 0, grid.width - 1)
    iy = np.clip(np.round(graph.node_pos[:, 1]).astype(int), 0, grid.height - 1)
    if impedance_species is None:
        dens = grid.grid[:, iy, ix].sum(axis=0)
    else:
        dens = np.where(
            impedance_species >= 0,
            grid.grid[impedance_species, iy, ix],
            grid.grid[:, iy, ix].sum(axis=0),
        )
    drag = 1.0 - dt * (damping + density_imped * dens)
    graph.node_vel *= np.clip(drag, 0.0, 1.0)[:, None]

    graph.node_pos += graph.node_vel * dt

    over_x0 = graph.node_pos[:, 0] < 0.2
    over_x1 = graph.node_pos[:, 0] > (w - 0.2)
    graph.node_pos[over_x0, 0] = 0.2
    graph.node_pos[over_x1, 0] = w - 0.2
    graph.node_vel[over_x0 | over_x1, 0] *= -1.0
    over_y0 = graph.node_pos[:, 1] < 0.2
    over_y1 = graph.node_pos[:, 1] > (h - 0.2)
    graph.node_pos[over_y0, 1] = 0.2
    graph.node_pos[over_y1, 1] = h - 0.2
    graph.node_vel[over_y0 | over_y1, 1] *= -1.0

    diff = graph.node_pos[:, None, :] - graph.node_pos[None, :, :]
    dist2 = (diff ** 2).sum(axis=2)
    min_dist = radius * 2.0
    min_dist2 = min_dist * min_dist
    mask = (dist2 > 1e-8) & (dist2 < min_dist2)
    if np.any(mask):
        dist = np.sqrt(dist2 + 1e-8)
        dir_vec = diff / dist[:, :, None]
        overlap = (min_dist - dist) * mask
        push = dir_vec * overlap[:, :, None] * 0.5
        disp = push.sum(axis=1) - push.sum(axis=0)
        graph.node_pos += disp

        relv = graph.node_vel[:, None, :] - graph.node_vel[None, :, :]
        vn = (relv * dir_vec).sum(axis=2)
        hit = (vn < 0.0) & mask
        impulse = (-0.5 * vn) * hit
        dv = dir_vec * impulse[:, :, None]
        graph.node_vel += dv.sum(axis=1) - dv.sum(axis=0)


def apply_node_torque(graph, dt, relax=0.08):
    if not graph.edges:
        return
    pos = graph.node_pos
    rest = graph.node_rest_dir
    stiff = graph.node_stiffness
    sum_dir = np.zeros_like(pos)
    for a, b in graph.edges:
        d = pos[b] - pos[a]
        norm = float(np.linalg.norm(d))
        if norm <= 1e-6:
            continue
        dn = d / norm
        sum_dir[a] += dn
        sum_dir[b] -= dn
    norm_sum = np.linalg.norm(sum_dir, axis=1)
    valid = norm_sum > 1e-6
    sum_dir[valid] = sum_dir[valid] / norm_sum[valid][:, None]
    rest = rest * (1.0 - relax) + sum_dir * relax
    rest_norm = np.linalg.norm(rest, axis=1)
    rest_valid = rest_norm > 1e-6
    rest[rest_valid] = rest[rest_valid] / rest_norm[rest_valid][:, None]
    graph.node_rest_dir = rest

    for a, b in graph.edges:
        d = pos[b] - pos[a]
        norm = float(np.linalg.norm(d))
        if norm <= 1e-6:
            continue
        dn = d / norm
        ra = graph.node_rest_dir[a]
        rb = graph.node_rest_dir[b]
        if np.linalg.norm(ra) > 1e-6:
            cross_a = ra[0] * dn[1] - ra[1] * dn[0]
            dot_a = ra[0] * dn[0] + ra[1] * dn[1]
            angle_a = math.atan2(cross_a, dot_a)
            perp_a = np.array([-dn[1], dn[0]], dtype=np.float32)
            torque_a = perp_a * (-angle_a) * stiff[a]
            graph.node_vel[a] += torque_a * dt
            graph.node_vel[b] -= torque_a * dt
        dn_b = -dn
        if np.linalg.norm(rb) > 1e-6:
            cross_b = rb[0] * dn_b[1] - rb[1] * dn_b[0]
            dot_b = rb[0] * dn_b[0] + rb[1] * dn_b[1]
            angle_b = math.atan2(cross_b, dot_b)
            perp_b = np.array([-dn_b[1], dn_b[0]], dtype=np.float32)
            torque_b = perp_b * (-angle_b) * stiff[b]
            graph.node_vel[b] += torque_b * dt
            graph.node_vel[a] -= torque_b * dt


class Graph:
    def __init__(self, species_count):
        self.species_count = species_count
        self.node_pos = np.zeros((0, 2), dtype=np.float32)
        self.node_vel = np.zeros((0, 2), dtype=np.float32)
        self.node_conc = np.zeros((0, species_count), dtype=np.float32)
        self.node_emit = np.zeros((0, species_count), dtype=np.float32)
        self.node_consume = np.zeros((0, species_count), dtype=np.float32)
        self.node_drive = np.ones((0, 3), dtype=np.float32)
        self.node_metric = np.zeros((0, 2, 2), dtype=np.float32)
        self.node_age = np.zeros(0, dtype=np.float32)
        self.node_phase = np.zeros(0, dtype=np.float32)
        self.node_stiffness = np.zeros(0, dtype=np.float32)
        self.node_rest_dir = np.zeros((0, 2), dtype=np.float32)
        self.edges = []
        self.edge_age = []
        self.node_filters = []
        self.node_kind = []
        self.sources = []

    def add_node(self, pos, initial=None, filters=None, kind="stem", drive_scale=None):
        pos = np.asarray(pos, dtype=np.float32)
        self.node_pos = np.vstack((self.node_pos, pos[None, :]))
        self.node_vel = np.vstack((self.node_vel, np.zeros((1, 2), dtype=np.float32)))
        if initial is None:
            initial = np.zeros(self.species_count, dtype=np.float32)
        self.node_conc = np.vstack((self.node_conc, initial[None, :]))
        self.node_emit = np.vstack((self.node_emit, np.zeros((1, self.species_count), dtype=np.float32)))
        self.node_consume = np.vstack((self.node_consume, np.zeros((1, self.species_count), dtype=np.float32)))
        if drive_scale is None:
            drive_scale = np.ones(3, dtype=np.float32)
        self.node_drive = np.vstack((self.node_drive, np.asarray(drive_scale, dtype=np.float32)[None, :]))
        self.node_metric = np.vstack((self.node_metric, np.zeros((1, 2, 2), dtype=np.float32)))
        self.node_age = np.append(self.node_age, 0.0)
        self.node_phase = np.append(self.node_phase, 0.0)
        self.node_stiffness = np.append(self.node_stiffness, 0.0)
        self.node_rest_dir = np.vstack((self.node_rest_dir, np.zeros((1, 2), dtype=np.float32)))
        self.node_filters.append(filters or [])
        self.node_kind.append(kind)
        return self.node_pos.shape[0] - 1

    def add_edge(self, a, b):
        if a == b:
            return
        if a > b:
            a, b = b, a
        if (a, b) not in self.edges:
            self.edges.append((a, b))
            self.edge_age.append(0.0)

    def remove_edge(self, edge_index):
        if 0 <= edge_index < len(self.edges):
            self.edges.pop(edge_index)
            self.edge_age.pop(edge_index)

    def add_source(self, node, species, rate):
        self.sources.append(NodeSource(node, species, rate))

    def apply_filters(self, dt):
        for i, filters in enumerate(self.node_filters):
            conc = self.node_conc[i]
            for flt in filters:
                conc = flt.apply(conc, dt)
            self.node_conc[i] = conc

    def apply_sources(self, dt):
        for src in self.sources:
            if 0 <= src.node < self.node_conc.shape[0]:
                self.node_conc[src.node, src.species] += src.rate * dt

    def diffuse(self, dt, diffusion):
        if not self.edges:
            return
        delta = np.zeros_like(self.node_conc)
        for a, b in self.edges:
            diff = self.node_conc[b] - self.node_conc[a]
            flux = diffusion * dt * diff
            delta[a] += flux
            delta[b] -= flux
        self.node_conc += delta

    def step(self, dt, diffusion):
        self.apply_sources(dt)
        self.diffuse(dt, diffusion)
        self.apply_filters(dt)
        self.node_conc = np.clip(self.node_conc, 0.0, 1.0)

    def exchange_with_grid(self, grid, rate, dt):
        if self.node_pos.size == 0:
            return
        for i, pos in enumerate(self.node_pos):
            x = int(round(pos[0]))
            y = int(round(pos[1]))
            if 0 <= x < grid.width and 0 <= y < grid.height:
                g = grid.grid[:, y, x]
                n = self.node_conc[i]
                flux = (g - n) * rate * dt
                self.node_conc[i] = n + flux
                grid.grid[:, y, x] = g - flux


def diffuse_grid(grid, diffusion, dt):
    padded = np.pad(grid, 1, mode="edge")
    lap = (
        padded[1:-1, 2:] + padded[1:-1, :-2]
        + padded[2:, 1:-1] + padded[:-2, 1:-1]
        - 4.0 * grid
    )
    return grid + diffusion * dt * lap


def hue_blend(conc, hues, gain=1.0, base_rgb=(180, 180, 175)):
    weights = np.maximum(conc, 0.0)
    total = weights.sum()
    if total <= 1e-6:
        return base_rgb
    angles = np.deg2rad(hues)
    vx = float((weights * np.cos(angles)).sum())
    vy = float((weights * np.sin(angles)).sum())
    hue = (math.atan2(vy, vx) / (2.0 * math.pi)) % 1.0
    mag = math.sqrt(vx * vx + vy * vy)
    sat = min(1.0, mag / (total + 1e-6))
    val = min(1.0, gain * total)
    rgb = hsv_to_rgb(hue, sat, val)
    alpha = max(0.0, min(1.0, total))
    br, bg, bb = base_rgb
    rr = int(br * (1.0 - alpha) + rgb[0] * alpha)
    rg = int(bg * (1.0 - alpha) + rgb[1] * alpha)
    rb = int(bb * (1.0 - alpha) + rgb[2] * alpha)
    return (rr, rg, rb)


def hsv_to_rgb(h, s, v):
    if s <= 1e-6:
        c = int(max(0.0, min(1.0, v)) * 255)
        return (c, c, c)
    h = (h % 1.0) * 6.0
    i = int(h)
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return (int(r * 255), int(g * 255), int(b * 255))


def draw_background(screen, grid_rect):
    width, height = screen.get_size()
    sky_top = np.array([205, 215, 225], dtype=np.float32)
    sky_bottom = np.array([225, 218, 205], dtype=np.float32)
    for y in range(height):
        t = y / max(1, height - 1)
        col = (sky_top * (1.0 - t) + sky_bottom * t).astype(np.uint8)
        screen.fill(col.tolist(), rect=pygame.Rect(0, y, width, 1))
    pygame.draw.rect(screen, (178, 178, 170), grid_rect)


def screen_pos_from_grid(origin, cell_size, pos):
    return (
        int(origin[0] + pos[0] * cell_size),
        int(origin[1] + pos[1] * cell_size),
    )


def compute_grid_jacobian(grid):
    species_count, height, width = grid.shape
    jacobian = np.zeros((species_count, height, width, 2), dtype=np.float32)
    for s in range(species_count):
        gy, gx = np.gradient(grid[s], axis=(0, 1))
        jacobian[s, :, :, 0] = gx
        jacobian[s, :, :, 1] = gy
    return jacobian


def sample_node_env(grid, pos):
    if pos.size == 0:
        return None
    x = int(np.clip(round(pos[0]), 0, grid.width - 1))
    y = int(np.clip(round(pos[1]), 0, grid.height - 1))
    metric_mean = {}
    for s in range(grid.species_count):
        metric_mean[s] = grid.sample_metric_mean(s, x, y)
    return {
        "grid_conc": grid.grid[:, y, x].copy(),
        "grid_xy": (x, y),
        "height": float(pos[1]) / max(1.0, grid.height - 1),
        "metric_mean": metric_mean,
    }


def default_speciation(env, current_kind):
    if env is None:
        return current_kind
    conc = env["grid_conc"]
    heat = float(conc[0]) if conc.shape[0] > 0 else 0.0
    nutrient = float(conc[1]) if conc.shape[0] > 1 else 0.0
    moisture = float(conc[2]) if conc.shape[0] > 2 else 0.0
    signal = float(conc[3]) if conc.shape[0] > 3 else 0.0
    agar = float(conc[4]) if conc.shape[0] > 4 else 0.0
    if agar > 0.5 and nutrient > 0.25:
        return "slime"
    if moisture > 0.55 and env["height"] > 0.45:
        return "water_root"
    if nutrient > 0.45 and env["height"] < 0.4:
        return "air_root"
    if heat > 0.5 or signal > 0.35:
        return "leaf"
    return "stem"


class World:
    def __init__(self, grid, graph, node_types, species_defs, speciate_fn=None):
        self.grid = grid
        self.graph = graph
        self.node_types = node_types
        self.species_defs = species_defs
        self.speciate_fn = speciate_fn or default_speciation
        self.node_radius = 0.32
        self.node_damping = 0.25
        self.node_impedance = 1.8
        self.drive_global = {"explore": 0.75, "exploit": 0.9, "wander": 0.25}
        self.drive_keys = ("explore", "exploit", "wander")
        self.rng = np.random.default_rng()
        self.species_index = {s.name: i for i, s in enumerate(species_defs)}
        self.kind_index = {name: i for i, name in enumerate(node_types.keys())}
        self.kind_emit = self._build_kind_profiles("emit")
        self.kind_consume = self._build_kind_profiles("consume")
        self.kind_motility = self._build_kind_motility()
        self.last_jacobian = None
        self.slime_index = self.kind_index.get("slime", -1)
        self.agar_index = self.species_index.get("agar", -1)
        self.signal_index = self.species_index.get("signal", -1)
        self.pellets = []
        self.global_period = 6.0
        self.global_phase = 0.0
        self.bud_threshold = 0.98
        self.bud_ready = False
        self.node_cycle_period = 10.0

    def add_pellet(self, pos, species_name, rate, radius=1.0):
        if species_name not in self.species_index:
            return
        self.pellets.append(
            {
                "pos": np.asarray(pos, dtype=np.float32),
                "species": self.species_index[species_name],
                "rate": float(rate),
                "radius": float(radius),
            }
        )

    def set_drive_global(self, explore=None, exploit=None, wander=None):
        if explore is not None:
            self.drive_global["explore"] = float(explore)
        if exploit is not None:
            self.drive_global["exploit"] = float(exploit)
        if wander is not None:
            self.drive_global["wander"] = float(wander)

    def set_node_drive(self, node_idx, explore=None, exploit=None, wander=None):
        if 0 <= node_idx < self.graph.node_drive.shape[0]:
            if explore is not None:
                self.graph.node_drive[node_idx, 0] = float(explore)
            if exploit is not None:
                self.graph.node_drive[node_idx, 1] = float(exploit)
            if wander is not None:
                self.graph.node_drive[node_idx, 2] = float(wander)

    def update_node_metrics(self):
        if self.graph.node_pos.size == 0 or self.slime_index < 0:
            return
        kind_idx = self._node_kind_indices()
        slime_mask = kind_idx == self.slime_index
        if not np.any(slime_mask):
            return
        v = self.graph.node_vel
        speed = np.linalg.norm(v, axis=1)
        dir_vec = np.zeros_like(v)
        valid = speed > 1e-6
        dir_vec[valid] = v[valid] / speed[valid][:, None]
        energy = np.zeros(self.graph.node_pos.shape[0], dtype=np.float32)
        if self.agar_index >= 0:
            energy = self.graph.node_conc[:, self.agar_index]
        scale = 0.2 + 0.8 * np.clip(energy, 0.0, 1.0)
        metric = np.zeros_like(self.graph.node_metric)
        metric[:, 0, 0] = dir_vec[:, 0] * dir_vec[:, 0]
        metric[:, 0, 1] = dir_vec[:, 0] * dir_vec[:, 1]
        metric[:, 1, 0] = dir_vec[:, 0] * dir_vec[:, 1]
        metric[:, 1, 1] = dir_vec[:, 1] * dir_vec[:, 1]
        metric *= scale[:, None, None]
        self.graph.node_metric[slime_mask] = metric[slime_mask]

    def _build_kind_profiles(self, attr):
        profiles = []
        for name in self.node_types.keys():
            profile = np.zeros(len(self.species_defs), dtype=np.float32)
            node_type = self.node_types[name]
            mapping = getattr(node_type, attr)
            if mapping:
                for sname, rate in mapping.items():
                    if sname in self.species_index:
                        profile[self.species_index[sname]] = float(rate)
            profiles.append(profile)
        return np.stack(profiles, axis=0)

    def _build_kind_motility(self):
        motility_species = np.full(len(self.node_types), -1, dtype=np.int32)
        motility_gain = np.zeros(len(self.node_types), dtype=np.float32)
        drive_weights = np.ones((len(self.node_types), 3), dtype=np.float32)
        for i, name in enumerate(self.node_types.keys()):
            node_type = self.node_types[name]
            if node_type.motility_species in self.species_index:
                motility_species[i] = self.species_index[node_type.motility_species]
            motility_gain[i] = float(node_type.motility_gain or 0.0)
            weights = node_type.drive_weights or {}
            drive_weights[i, 0] = float(weights.get("explore", 1.0))
            drive_weights[i, 1] = float(weights.get("exploit", 1.0))
            drive_weights[i, 2] = float(weights.get("wander", 1.0))
        return {
            "species": motility_species,
            "gain": motility_gain,
            "drive_weights": drive_weights,
        }

    def _node_kind_indices(self):
        if not self.graph.node_kind:
            return np.zeros(0, dtype=np.int32)
        return np.array([self.kind_index.get(k, 0) for k in self.graph.node_kind], dtype=np.int32)

    def apply_node_flux(self, dt):
        if self.graph.node_pos.size == 0:
            return
        ix = np.clip(np.round(self.graph.node_pos[:, 0]).astype(int), 0, self.grid.width - 1)
        iy = np.clip(np.round(self.graph.node_pos[:, 1]).astype(int), 0, self.grid.height - 1)
        kind_idx = self._node_kind_indices()
        emit = self.kind_emit[kind_idx]
        consume = self.kind_consume[kind_idx]
        self.graph.node_emit = emit
        self.graph.node_consume = consume
        if emit.any():
            for s in range(self.grid.species_count):
                if emit[:, s].any():
                    add = emit[:, s] * dt
                    np.add.at(self.grid.grid[s], (iy, ix), add)
                    if self.species_defs[s].metric_enabled:
                        add_emit = np.zeros((self.grid.height, self.grid.width), dtype=np.float32)
                        np.add.at(add_emit, (iy, ix), add)
                        add_metric = np.zeros((2, 2, self.grid.height, self.grid.width), dtype=np.float32)
                        for i in range(2):
                            for j in range(2):
                                np.add.at(
                                    add_metric[i, j],
                                    (iy, ix),
                                    add * self.graph.node_metric[:, i, j],
                                )
                        conc = self.grid.grid[s]
                        metric = self.grid.metric[s]
                        denom = conc + add_emit
                        safe = denom > 1e-6
                        for i in range(2):
                            for j in range(2):
                                blended = np.zeros_like(metric[i, j])
                                blended[safe] = (
                                    metric[i, j][safe] * conc[safe] + add_metric[i, j][safe]
                                ) / denom[safe]
                                metric[i, j] = blended
        if consume.any():
            for s in range(self.grid.species_count):
                if consume[:, s].any():
                    np.add.at(self.grid.grid[s], (iy, ix), -consume[:, s] * dt)
        self.grid.grid = np.clip(self.grid.grid, 0.0, 1.0)

    def apply_pellets(self, dt):
        if not self.pellets:
            return
        for pellet in self.pellets:
            x, y = pellet["pos"]
            s = pellet["species"]
            r = pellet["radius"]
            rate = pellet["rate"]
            x0 = max(0, int(round(x - r)))
            x1 = min(self.grid.width - 1, int(round(x + r)))
            y0 = max(0, int(round(y - r)))
            y1 = min(self.grid.height - 1, int(round(y + r)))
            if x1 < x0 or y1 < y0:
                continue
            xs = np.arange(x0, x1 + 1)
            ys = np.arange(y0, y1 + 1)
            gx, gy = np.meshgrid(xs, ys)
            dx = gx.astype(np.float32) - float(x)
            dy = gy.astype(np.float32) - float(y)
            mask = (dx * dx + dy * dy) <= (r * r)
            if np.any(mask):
                self.grid.grid[s, y0 : y1 + 1, x0 : x1 + 1][mask] += rate * dt

    def apply_motility(self, dt):
        if self.graph.node_pos.size == 0:
            return
        jac = self.last_jacobian
        if jac is None:
            return
        kind_idx = self._node_kind_indices()
        mot_species = self.kind_motility["species"][kind_idx]
        mot_gain = self.kind_motility["gain"][kind_idx]
        drive_weights = self.kind_motility["drive_weights"][kind_idx]
        if not np.any(mot_gain > 0.0):
            return
        ix = np.clip(np.round(self.graph.node_pos[:, 0]).astype(int), 0, self.grid.width - 1)
        iy = np.clip(np.round(self.graph.node_pos[:, 1]).astype(int), 0, self.grid.height - 1)
        grad = np.zeros((self.graph.node_pos.shape[0], 2), dtype=np.float32)
        mask = mot_species >= 0
        if np.any(mask):
            grad[mask] = jac[mot_species[mask], iy[mask], ix[mask]]
        grad_norm = np.linalg.norm(grad, axis=1)
        grad_dir = np.zeros_like(grad)
        valid = grad_norm > 1e-6
        grad_dir[valid] = grad[valid] / grad_norm[valid][:, None]

        random_dir = self.rng.uniform(-1.0, 1.0, size=grad.shape).astype(np.float32)
        rnd_norm = np.linalg.norm(random_dir, axis=1)
        rnd_valid = rnd_norm > 1e-6
        random_dir[rnd_valid] = random_dir[rnd_valid] / rnd_norm[rnd_valid][:, None]

        drive_scale = self.graph.node_drive
        explore = self.drive_global["explore"] * drive_weights[:, 0] * drive_scale[:, 0]
        exploit = self.drive_global["exploit"] * drive_weights[:, 1] * drive_scale[:, 1]
        wander = self.drive_global["wander"] * drive_weights[:, 2] * drive_scale[:, 2]

        agar_idx = self.species_index.get("agar", -1)
        if agar_idx >= 0:
            local_agar = self.grid.grid[agar_idx, iy, ix]
            exploit = exploit * local_agar
            explore = explore * (1.0 - local_agar)

        impulse = (
            grad_dir * exploit[:, None]
            + random_dir * explore[:, None]
            + random_dir * wander[:, None]
        )
        self.graph.node_vel += impulse * mot_gain[:, None] * dt

    def _impedance_indices(self):
        if not self.graph.node_kind:
            return None
        out = np.full(len(self.graph.node_kind), -1, dtype=np.int32)
        for i, kind in enumerate(self.graph.node_kind):
            node_type = self.node_types.get(kind)
            if node_type and node_type.impedance_species is not None:
                out[i] = int(node_type.impedance_species)
        return out

    def step(self, dt, grid_diffusion, graph_diffusion, coupling_rate):
        self.graph.node_age += dt
        self.graph.node_phase = (self.graph.node_phase + dt / self.node_cycle_period) % 1.0
        if self.graph.edge_age:
            for i in range(len(self.graph.edge_age)):
                self.graph.edge_age[i] += dt
        prev_phase = self.global_phase
        self.global_phase = (self.global_phase + dt / self.global_period) % 1.0
        if prev_phase < self.bud_threshold <= self.global_phase:
            self.bud_ready = True
        self.update_node_metrics()
        self.update_node_properties()
        self.apply_pellets(dt)
        self.apply_node_flux(dt)
        self.grid.step(dt, grid_diffusion, self.species_defs)
        self.graph.step(dt, graph_diffusion)
        self.graph.exchange_with_grid(self.grid, coupling_rate, dt)
        self.last_jacobian = compute_grid_jacobian(self.grid.grid)
        for i, kind in enumerate(self.graph.node_kind):
            node_type = self.node_types.get(kind)
            if node_type:
                node_type.on_exchange(self, i, dt)
        for i, kind in enumerate(self.graph.node_kind):
            node_type = self.node_types.get(kind)
            if node_type:
                node_type.on_pre_step(self, i, dt)
        self.apply_motility(dt)
        apply_node_torque(self.graph, dt)
        step_node_physics(
            self.graph,
            self.grid,
            dt,
            (self.grid.width - 1, self.grid.height - 1),
            damping=self.node_damping,
            density_imped=self.node_impedance,
            radius=self.node_radius,
            impedance_species=self._impedance_indices(),
        )
        for i, kind in enumerate(self.graph.node_kind):
            node_type = self.node_types.get(kind)
            if node_type:
                node_type.on_post_step(self, i, dt)
        for i, kind in enumerate(self.graph.node_kind):
            env = sample_node_env(self.grid, self.graph.node_pos[i])
            new_kind = kind
            node_type = self.node_types.get(kind)
            if node_type:
                new_kind = node_type.on_speciate(env, kind)
            if new_kind == kind:
                new_kind = self.speciate_fn(env, kind)
            if new_kind != kind and new_kind in self.node_types:
                self.graph.node_kind[i] = new_kind

    def update_node_properties(self):
        if not self.graph.node_kind:
            return
        for i, kind in enumerate(self.graph.node_kind):
            node_type = self.node_types.get(kind)
            if node_type:
                self.graph.node_stiffness[i] = float(node_type.stiffness)

    def consume_bud_ready(self):
        if self.bud_ready:
            self.bud_ready = False
            return True
        return False


def select_growth_edge(graph, jacobian, rule):
    if not graph.edges:
        return None
    if jacobian is None or graph.node_pos.size == 0:
        candidates = list(range(len(graph.edges)))
    else:
        candidates = list(range(len(graph.edges)))
    if rule.applicable_kinds != ("any", "any"):
        filtered = []
        for idx in candidates:
            a, b = graph.edges[idx]
            ka = graph.node_kind[a]
            kb = graph.node_kind[b]
            if (ka, kb) == rule.applicable_kinds or (kb, ka) == rule.applicable_kinds:
                filtered.append(idx)
        candidates = filtered
    if not candidates:
        return None
    if jacobian is None or graph.node_pos.size == 0:
        return random.choice(candidates)
    height = jacobian.shape[1]
    width = jacobian.shape[2]
    best_idx = None
    best_score = -1e9
    for idx in candidates:
        a, b = graph.edges[idx]
        pa = graph.node_pos[a]
        pb = graph.node_pos[b]
        mid = (pa + pb) * 0.5
        x = int(round(mid[0]))
        y = int(round(mid[1]))
        if not (0 <= x < width and 0 <= y < height):
            continue
        weights = (graph.node_conc[a] + graph.node_conc[b]) * 0.5
        grad = jacobian[:, y, x, :]
        combined = (grad * weights[:, None]).sum(axis=0)
        edge_dir = pb - pa
        length = float(np.linalg.norm(edge_dir))
        if length <= 1e-6:
            continue
        edge_dir = edge_dir / length
        alignment = abs(float(np.dot(combined, edge_dir)))
        score = alignment + 0.25 * float(np.linalg.norm(combined))
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx is None:
        return random.choice(candidates)
    return best_idx


def apply_subgraph_rule(graph, rule, edge_index, max_nodes):
    if edge_index < 0 or edge_index >= len(graph.edges):
        return False
    if graph.node_pos.shape[0] + len(rule.new_nodes) > max_nodes:
        return False
    a, b = graph.edges[edge_index]
    start_kind = graph.node_kind[a]
    end_kind = graph.node_kind[b]
    if rule.applicable_kinds != ("any", "any"):
        if (start_kind, end_kind) != rule.applicable_kinds and (end_kind, start_kind) != rule.applicable_kinds:
            return False
    p0 = graph.node_pos[a]
    p1 = graph.node_pos[b]
    edge_vec = p1 - p0
    length = float(np.linalg.norm(edge_vec))
    if length <= 1e-6:
        return False
    tangent = edge_vec / length
    perp = np.array([-tangent[1], tangent[0]], dtype=np.float32)
    new_indices = []
    for (u, v), kind in zip(rule.new_nodes, rule.new_kinds):
        pos = p0 + edge_vec * float(u) + perp * float(v) * length
        initial = (graph.node_conc[a] + graph.node_conc[b]) * 0.5
        idx = graph.add_node(
            pos,
            initial=initial.copy(),
            filters=[ClampFilter(0.0, 1.0)],
            kind=kind,
        )
        new_indices.append(idx)
    graph.remove_edge(edge_index)
    mapping = [a, b] + new_indices
    for ea, eb in rule.edges:
        graph.add_edge(mapping[ea], mapping[eb])
    return True


def build_rules():
    return [
        SubgraphRule(
            name="slime_extend",
            applicable_kinds=("slime", "slime"),
            new_nodes=[(0.5, 0.0)],
            edges=[(0, 2), (2, 1)],
            new_kinds=["slime"],
        ),
        SubgraphRule(
            name="slime_branch",
            applicable_kinds=("slime", "slime"),
            new_nodes=[(0.5, 0.25), (0.5, -0.25)],
            edges=[(0, 2), (0, 3)],
            new_kinds=["slime", "slime"],
        ),
        SubgraphRule(
            name="stem_elongate",
            applicable_kinds=("stem", "stem"),
            new_nodes=[(0.5, 0.0)],
            edges=[(0, 2), (2, 1)],
            new_kinds=["stem"],
        ),
        SubgraphRule(
            name="leaf_lateral",
            applicable_kinds=("stem", "stem"),
            new_nodes=[(0.6, 0.3)],
            edges=[(0, 2), (2, 1)],
            new_kinds=["leaf"],
        ),
        SubgraphRule(
            name="root_split",
            applicable_kinds=("water_root", "water_root"),
            new_nodes=[(0.5, 0.2), (0.5, -0.2)],
            edges=[(0, 2), (0, 3)],
            new_kinds=["water_root", "water_root"],
        ),
        SubgraphRule(
            name="air_root_drop",
            applicable_kinds=("stem", "stem"),
            new_nodes=[(0.4, -0.25)],
            edges=[(0, 2)],
            new_kinds=["air_root"],
        ),
    ]


def make_initial_graph(species_count):
    graph = Graph(species_count)
    a = graph.add_node(
        (10.0, 10.0),
        initial=np.array([0.2, 0.1, 0.0, 0.0, 0.0], dtype=np.float32),
        kind="slime",
    )
    b = graph.add_node(
        (18.0, 12.0),
        initial=np.array([0.0, 0.4, 0.0, 0.0, 0.0], dtype=np.float32),
        kind="slime",
    )
    c = graph.add_node(
        (26.0, 16.0),
        initial=np.array([0.0, 0.0, 0.3, 0.0, 0.0], dtype=np.float32),
        kind="slime",
    )
    d = graph.add_node(
        (14.0, 18.0),
        initial=np.array([0.0, 0.2, 0.1, 0.0, 0.3], dtype=np.float32),
        kind="slime",
        drive_scale=np.array([1.2, 1.0, 0.6], dtype=np.float32),
    )
    graph.add_edge(a, b)
    graph.add_edge(b, c)
    graph.add_edge(b, d)
    graph.node_filters[a].append(DecayFilter(0.08))
    graph.node_filters[b].append(ConvertFilter(1, 2, 0.2))
    graph.node_filters[c].append(ClampFilter(0.0, 1.0))
    graph.add_source(a, 0, 0.15)
    graph.add_source(c, 2, 0.08)
    return graph


def main():
    pygame.init()
    pygame.display.set_caption("Voxel + Graph Flux Harness")

    species_defs = [
        Species("heat", 20.0, 0.02),
        Species("nutrient", 120.0, 0.01),
        Species("moisture", 210.0, 0.01),
        Species("signal", 300.0, 0.02, metric_enabled=True, metric_decay=0.05, metric_diffusion=0.25),
        Species("agar", 70.0, 0.005),
    ]
    species_hues = np.array([s.hue_deg for s in species_defs], dtype=np.float32)

    grid_w = 40
    grid_h = 24
    cell = 20
    margin = 30
    screen_w = grid_w * cell + margin * 2
    screen_h = grid_h * cell + margin * 2

    screen = pygame.display.set_mode((screen_w, screen_h))
    font = pygame.font.SysFont("consolas", 16)

    grid = VoxelGrid(grid_w, grid_h, len(species_defs))
    grid.add_source(4, 4, 0, 0.2)
    grid.add_source(grid_w - 5, 6, 1, 0.12)
    grid.add_source(8, grid_h - 6, 2, 0.18)
    grid.add_blob(16, 14, 3, 0.3, radius=2)
    if len(species_defs) > 4:
        grid.grid[4, :, :] = 0.6

    graph = make_initial_graph(len(species_defs))
    rules = build_rules()
    rule_index = 0
    node_types = {
        "water_root": WaterRoot(),
        "air_root": AirRoot(),
        "stem": Stem(),
        "leaf": Leaf(),
        "slime": Slime(),
    }
    world = World(grid, graph, node_types, species_defs)
    world.add_pellet((6.0, 5.0), "agar", 0.05, radius=1.4)
    world.add_pellet((20.0, 6.0), "agar", 0.05, radius=1.4)
    world.add_pellet((32.0, 10.0), "agar", 0.05, radius=1.4)
    world.add_pellet((12.0, 18.0), "agar", 0.05, radius=1.4)

    grid_diffusion = 0.45
    graph_diffusion = 0.8
    coupling_rate = 0.6
    vis_gain = 1.0
    dt = 0.1
    max_nodes = 80

    clock = pygame.time.Clock()
    running = True
    while running:
        grow_requested = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    grow_requested = True
                if event.key == pygame.K_TAB:
                    rule_index = (rule_index + 1) % len(rules)
                if event.key == pygame.K_g:
                    grid_diffusion = min(2.0, grid_diffusion + 0.05)
                if event.key == pygame.K_h:
                    grid_diffusion = max(0.0, grid_diffusion - 0.05)
                if event.key == pygame.K_j:
                    graph_diffusion = min(3.0, graph_diffusion + 0.1)
                if event.key == pygame.K_k:
                    graph_diffusion = max(0.0, graph_diffusion - 0.1)
                if event.key == pygame.K_c:
                    coupling_rate = min(2.0, coupling_rate + 0.05)
                if event.key == pygame.K_v:
                    coupling_rate = max(0.0, coupling_rate - 0.05)
                if event.key == pygame.K_r:
                    grid = VoxelGrid(grid_w, grid_h, len(species_defs))
                    grid.add_source(4, 4, 0, 0.2)
                    grid.add_source(grid_w - 5, 6, 1, 0.12)
                    grid.add_source(8, grid_h - 6, 2, 0.18)
                    grid.add_blob(16, 14, 3, 0.3, radius=2)
                    if len(species_defs) > 4:
                        grid.grid[4, :, :] = 0.6
                    graph = make_initial_graph(len(species_defs))
                    world = World(grid, graph, node_types, species_defs)
                    world.add_pellet((6.0, 5.0), "agar", 0.05, radius=1.4)
                    world.add_pellet((20.0, 6.0), "agar", 0.05, radius=1.4)
                    world.add_pellet((32.0, 10.0), "agar", 0.05, radius=1.4)
                    world.add_pellet((12.0, 18.0), "agar", 0.05, radius=1.4)

        world.step(dt, grid_diffusion, graph_diffusion, coupling_rate)
        jacobian = world.last_jacobian
        if grow_requested and world.graph.edges:
            edge_index = select_growth_edge(world.graph, jacobian, rules[rule_index])
            if edge_index is not None:
                apply_subgraph_rule(world.graph, rules[rule_index], edge_index, max_nodes)
        if world.consume_bud_ready() and world.graph.edges:
            edge_index = select_growth_edge(world.graph, jacobian, rules[rule_index])
            if edge_index is not None:
                apply_subgraph_rule(world.graph, rules[rule_index], edge_index, max_nodes)

        grid_rect = pygame.Rect(margin, margin, grid_w * cell, grid_h * cell)
        draw_background(screen, grid_rect)

        grid_base = (170, 170, 162)
        edge_base = (205, 205, 205)
        node_base = (220, 220, 220)
        for y in range(grid_h):
            for x in range(grid_w):
                conc = world.grid.grid[:, y, x]
                rgb = hue_blend(conc, species_hues, gain=vis_gain, base_rgb=grid_base)
                pygame.draw.rect(
                    screen,
                    rgb,
                    pygame.Rect(
                        margin + x * cell,
                        margin + y * cell,
                        cell - 1,
                        cell - 1,
                    ),
                )

        line_width = 6
        knob_radius = line_width // 2 + 1
        for a, b in world.graph.edges:
            pa = world.graph.node_pos[a]
            pb = world.graph.node_pos[b]
            ca = world.graph.node_conc[a]
            cb = world.graph.node_conc[b]
            rgb = hue_blend((ca + cb) * 0.5, species_hues, gain=vis_gain, base_rgb=edge_base)
            pygame.draw.line(
                screen,
                rgb,
                screen_pos_from_grid((margin, margin), cell, pa),
                screen_pos_from_grid((margin, margin), cell, pb),
                line_width,
            )

        for i, pos in enumerate(world.graph.node_pos):
            rgb = hue_blend(world.graph.node_conc[i], species_hues, gain=vis_gain, base_rgb=node_base)
            pygame.draw.circle(
                screen,
                rgb,
                screen_pos_from_grid((margin, margin), cell, pos),
                knob_radius,
            )

        hud = (
            f"rule:{rules[rule_index].name}  nodes:{world.graph.node_pos.shape[0]}  edges:{len(world.graph.edges)}  "
            f"gridD:{grid_diffusion:.2f} graphD:{graph_diffusion:.2f} coupling:{coupling_rate:.2f}"
        )
        hud2 = "SPACE:grow  TAB:rule  G/H: grid diff  J/K: graph diff  C/V: coupling  R:reset"
        text1 = font.render(hud, True, (20, 20, 20))
        text2 = font.render(hud2, True, (20, 20, 20))
        screen.blit(text1, (margin, 6))
        screen.blit(text2, (margin, screen_h - 22))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
