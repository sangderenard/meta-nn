import math
from dataclasses import dataclass

import torch
import pygame


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this simulation.")
    return torch.device("cuda")


def laplacian_2d(field):
    if field.dim() == 2:
        left = field[:, :1]
        right = field[:, -1:]
        field = torch.cat([left, field, right], dim=1)
        top = field[:1, :]
        bottom = field[-1:, :]
        padded = torch.cat([top, field, bottom], dim=0)
        center = padded[1:-1, 1:-1]
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]
        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        return left + right + up + down - 4.0 * center
    left = field[..., :, :1]
    right = field[..., :, -1:]
    field = torch.cat([left, field, right], dim=-1)
    top = field[..., :1, :]
    bottom = field[..., -1:, :]
    padded = torch.cat([top, field, bottom], dim=-2)
    center = padded[..., 1:-1, 1:-1]
    left = padded[..., 1:-1, :-2]
    right = padded[..., 1:-1, 2:]
    up = padded[..., :-2, 1:-1]
    down = padded[..., 2:, 1:-1]
    return left + right + up + down - 4.0 * center


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


def hue_blend(conc, hues, gain=1.0, base_rgb=(175, 175, 170)):
    weights = conc.clip(min=0.0)
    total = weights.sum()
    if total <= 1e-6:
        return base_rgb
    angles = torch.deg2rad(hues)
    vx = float((weights * torch.cos(angles)).sum())
    vy = float((weights * torch.sin(angles)).sum())
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


@dataclass
class Species:
    name: str
    hue_deg: float
    decay: float
    diffusion: float = 0.4
    metric_enabled: bool = False
    metric_decay: float = 0.0
    metric_diffusion: float = 0.15

    def step_concentration(self, grid, dt):
        diffused = grid + self.diffusion * dt * laplacian_2d(grid)
        if self.decay > 0.0:
            diffused = diffused * (1.0 - self.decay * dt)
        return diffused

    def step_metric(self, metric, dt):
        if not self.metric_enabled:
            return metric
        diffused = metric + self.metric_diffusion * dt * laplacian_2d(metric)
        if self.metric_decay > 0.0:
            diffused = diffused * (1.0 - self.metric_decay * dt)
        return diffused


class VoxelGrid:
    def __init__(self, width, height, species_defs, device):
        self.width = width
        self.height = height
        self.species_defs = species_defs
        self.species_count = len(species_defs)
        self.device = device
        self.grid = torch.zeros(
            (self.species_count, height, width),
            device=device,
            dtype=torch.float32,
        )
        self.metric = torch.zeros(
            (self.species_count, 2, 2, height, width),
            device=device,
            dtype=torch.float32,
        )

    def step(self, dt):
        for s, spec in enumerate(self.species_defs):
            self.grid[s] = spec.step_concentration(self.grid[s], dt).clamp_(0.0, 1.0)
            if spec.metric_enabled:
                for i in range(2):
                    for j in range(2):
                        self.metric[s, i, j] = spec.step_metric(self.metric[s, i, j], dt)
                low = self.grid[s] <= 1e-6
                if low.any():
                    self.metric[s, :, :, low] = 0.0

    def sample_metric_mean(self, species_idx, x, y, eps=1e-6):
        conc = self.grid[species_idx, y, x]
        if float(conc) <= eps:
            return torch.zeros((2, 2), device=self.device)
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
    metabolism_rate = 0.0

    def on_pre_step(self, world, node_idx, dt):
        pass

    def on_post_step(self, world, node_idx, dt):
        pass

    def on_exchange(self, world, node_idx, dt):
        pass

    def on_speciate(self, env, current_kind):
        return current_kind


class Slime(NodeType):
    name = "slime"
    impedance_species = "agar"
    emit = {"signal": 0.06}
    consume = {"agar": 0.04}
    motility_species = "agar"
    motility_gain = 1.2
    drive_weights = {"explore": 1.2, "exploit": 1.4, "wander": 0.35}
    stiffness = 0.08
    metabolism_rate = 0.06

    def on_pre_step(self, world, node_idx, dt):
        agar = world.graph.node_conc[node_idx, world.species_index["agar"]]
        burn = torch.minimum(agar, torch.tensor(self.metabolism_rate * dt, device=world.device))
        world.graph.node_conc[node_idx, world.species_index["agar"]] = agar - burn


class Graph:
    def __init__(self, species_count, device):
        self.device = device
        self.species_count = species_count
        self.node_pos = torch.zeros((0, 2), device=device)
        self.node_vel = torch.zeros((0, 2), device=device)
        self.node_conc = torch.zeros((0, species_count), device=device)
        self.node_emit = torch.zeros((0, species_count), device=device)
        self.node_consume = torch.zeros((0, species_count), device=device)
        self.node_drive = torch.ones((0, 3), device=device)
        self.node_metric = torch.zeros((0, 2, 2), device=device)
        self.node_age = torch.zeros((0,), device=device)
        self.node_phase = torch.zeros((0,), device=device)
        self.node_stiffness = torch.zeros((0,), device=device)
        self.node_rest_dir = torch.zeros((0, 2), device=device)
        self.node_kind = []
        self.edges = torch.zeros((0, 2), dtype=torch.long, device=device)
        self.edge_age = torch.zeros((0,), device=device)

    def add_node(self, pos, initial=None, kind="slime", drive_scale=None):
        pos = torch.tensor(pos, device=self.device, dtype=torch.float32).view(1, 2)
        self.node_pos = torch.cat([self.node_pos, pos], dim=0)
        self.node_vel = torch.cat([self.node_vel, torch.zeros((1, 2), device=self.device)], dim=0)
        if initial is None:
            initial = torch.zeros((1, self.species_count), device=self.device)
        else:
            initial = torch.tensor(initial, device=self.device, dtype=torch.float32).view(1, -1)
        self.node_conc = torch.cat([self.node_conc, initial], dim=0)
        self.node_emit = torch.cat(
            [self.node_emit, torch.zeros((1, self.species_count), device=self.device)], dim=0
        )
        self.node_consume = torch.cat(
            [self.node_consume, torch.zeros((1, self.species_count), device=self.device)], dim=0
        )
        if drive_scale is None:
            drive_scale = torch.ones((1, 3), device=self.device)
        else:
            drive_scale = torch.tensor(drive_scale, device=self.device, dtype=torch.float32).view(1, 3)
        self.node_drive = torch.cat([self.node_drive, drive_scale], dim=0)
        self.node_metric = torch.cat([self.node_metric, torch.zeros((1, 2, 2), device=self.device)], dim=0)
        self.node_age = torch.cat([self.node_age, torch.zeros((1,), device=self.device)], dim=0)
        self.node_phase = torch.cat([self.node_phase, torch.zeros((1,), device=self.device)], dim=0)
        self.node_stiffness = torch.cat([self.node_stiffness, torch.zeros((1,), device=self.device)], dim=0)
        self.node_rest_dir = torch.cat([self.node_rest_dir, torch.zeros((1, 2), device=self.device)], dim=0)
        self.node_kind.append(kind)
        return self.node_pos.shape[0] - 1

    def add_edge(self, a, b):
        if a == b:
            return
        if a > b:
            a, b = b, a
        edge = torch.tensor([[a, b]], device=self.device, dtype=torch.long)
        if self.edges.shape[0] == 0:
            self.edges = edge
            self.edge_age = torch.zeros((1,), device=self.device)
            return
        exists = ((self.edges[:, 0] == edge[0, 0]) & (self.edges[:, 1] == edge[0, 1])).any()
        if not bool(exists):
            self.edges = torch.cat([self.edges, edge], dim=0)
            self.edge_age = torch.cat([self.edge_age, torch.zeros((1,), device=self.device)], dim=0)


@dataclass
class Pellet:
    pos: torch.Tensor
    species: int
    rate: float
    radius: float


class World:
    def __init__(self, grid, graph, node_types, device):
        self.grid = grid
        self.graph = graph
        self.node_types = node_types
        self.device = device
        self.species_index = {s.name: i for i, s in enumerate(grid.species_defs)}
        self.kind_index = {name: i for i, name in enumerate(node_types.keys())}
        self.global_period = 6.0
        self.global_phase = 0.0
        self.bud_threshold = 0.98
        self.bud_ready = False
        self.node_cycle_period = 10.0
        self.drive_global = torch.tensor([0.75, 0.9, 0.25], device=device)
        self.rng = torch.Generator(device=device)
        self.pellets = []
        self.last_jacobian = None
        self._build_kind_profiles()

    def _build_kind_profiles(self):
        kind_count = len(self.node_types)
        species_count = self.grid.species_count
        emit = torch.zeros((kind_count, species_count), device=self.device)
        consume = torch.zeros((kind_count, species_count), device=self.device)
        mot_species = torch.full((kind_count,), -1, device=self.device, dtype=torch.long)
        mot_gain = torch.zeros((kind_count,), device=self.device)
        drive_weights = torch.ones((kind_count, 3), device=self.device)
        stiffness = torch.zeros((kind_count,), device=self.device)
        for name, idx in self.kind_index.items():
            ntype = self.node_types[name]
            if ntype.emit:
                for sname, rate in ntype.emit.items():
                    emit[idx, self.species_index[sname]] = float(rate)
            if ntype.consume:
                for sname, rate in ntype.consume.items():
                    consume[idx, self.species_index[sname]] = float(rate)
            if ntype.motility_species:
                mot_species[idx] = self.species_index[ntype.motility_species]
            mot_gain[idx] = float(ntype.motility_gain)
            weights = ntype.drive_weights or {}
            drive_weights[idx, 0] = float(weights.get("explore", 1.0))
            drive_weights[idx, 1] = float(weights.get("exploit", 1.0))
            drive_weights[idx, 2] = float(weights.get("wander", 1.0))
            stiffness[idx] = float(ntype.stiffness)
        self.kind_emit = emit
        self.kind_consume = consume
        self.kind_motility_species = mot_species
        self.kind_motility_gain = mot_gain
        self.kind_drive_weights = drive_weights
        self.kind_stiffness = stiffness

    def add_pellet(self, pos, species_name, rate, radius=1.0):
        if species_name not in self.species_index:
            return
        self.pellets.append(
            Pellet(
                pos=torch.tensor(pos, device=self.device, dtype=torch.float32),
                species=self.species_index[species_name],
                rate=float(rate),
                radius=float(radius),
            )
        )

    def _node_kind_indices(self):
        if not self.graph.node_kind:
            return torch.zeros((0,), device=self.device, dtype=torch.long)
        idx = torch.tensor([self.kind_index[k] for k in self.graph.node_kind], device=self.device)
        return idx

    def apply_pellets(self, dt):
        if not self.pellets:
            return
        for pellet in self.pellets:
            x, y = pellet.pos
            r = pellet.radius
            x0 = int(max(0, torch.floor(x - r).item()))
            x1 = int(min(self.grid.width - 1, torch.ceil(x + r).item()))
            y0 = int(max(0, torch.floor(y - r).item()))
            y1 = int(min(self.grid.height - 1, torch.ceil(y + r).item()))
            xs = torch.arange(x0, x1 + 1, device=self.device)
            ys = torch.arange(y0, y1 + 1, device=self.device)
            gx, gy = torch.meshgrid(xs, ys, indexing="xy")
            dx = gx.float() - x
            dy = gy.float() - y
            mask = (dx * dx + dy * dy) <= (r * r)
            add = pellet.rate * dt
            patch = self.grid.grid[pellet.species, y0 : y1 + 1, x0 : x1 + 1]
            patch[mask] = patch[mask] + add

    def update_node_metrics(self):
        if self.graph.node_pos.shape[0] == 0:
            return
        v = self.graph.node_vel
        speed = torch.norm(v, dim=1, keepdim=True)
        dir_vec = torch.zeros_like(v)
        valid = speed.squeeze() > 1e-6
        dir_vec[valid] = v[valid] / speed[valid]
        energy = self.graph.node_conc[:, self.species_index["agar"]]
        scale = 0.2 + 0.8 * torch.clamp(energy, 0.0, 1.0)
        metric = torch.zeros_like(self.graph.node_metric)
        metric[:, 0, 0] = dir_vec[:, 0] * dir_vec[:, 0]
        metric[:, 0, 1] = dir_vec[:, 0] * dir_vec[:, 1]
        metric[:, 1, 0] = dir_vec[:, 0] * dir_vec[:, 1]
        metric[:, 1, 1] = dir_vec[:, 1] * dir_vec[:, 1]
        metric = metric * scale.view(-1, 1, 1)
        self.graph.node_metric = metric

    def apply_node_flux(self, dt):
        if self.graph.node_pos.shape[0] == 0:
            return
        kind_idx = self._node_kind_indices()
        emit = self.kind_emit[kind_idx]
        consume = self.kind_consume[kind_idx]
        self.graph.node_emit = emit
        self.graph.node_consume = consume
        ix = torch.clamp(self.graph.node_pos[:, 0].round().long(), 0, self.grid.width - 1)
        iy = torch.clamp(self.graph.node_pos[:, 1].round().long(), 0, self.grid.height - 1)
        flat = iy * self.grid.width + ix
        for s in range(self.grid.species_count):
            if emit[:, s].abs().sum() > 0:
                add = emit[:, s] * dt
                grid_flat = self.grid.grid[s].view(-1)
                grid_flat.scatter_add_(0, flat, add)
                if self.grid.species_defs[s].metric_enabled:
                    for i in range(2):
                        for j in range(2):
                            metric_flat = self.grid.metric[s, i, j].view(-1)
                            metric_flat.scatter_add_(0, flat, add * self.graph.node_metric[:, i, j])
            if consume[:, s].abs().sum() > 0:
                sub = consume[:, s] * dt
                grid_flat = self.grid.grid[s].view(-1)
                grid_flat.scatter_add_(0, flat, -sub)
        self.grid.grid.clamp_(0.0, 1.0)

    def apply_motility(self, dt):
        if self.last_jacobian is None or self.graph.node_pos.shape[0] == 0:
            return
        kind_idx = self._node_kind_indices()
        mot_species = self.kind_motility_species[kind_idx]
        mot_gain = self.kind_motility_gain[kind_idx]
        drive_weights = self.kind_drive_weights[kind_idx]
        if mot_gain.max().item() <= 0.0:
            return
        ix = torch.clamp(self.graph.node_pos[:, 0].round().long(), 0, self.grid.width - 1)
        iy = torch.clamp(self.graph.node_pos[:, 1].round().long(), 0, self.grid.height - 1)
        grad = torch.zeros_like(self.graph.node_pos)
        valid = mot_species >= 0
        if valid.any():
            grad[valid] = self.last_jacobian[mot_species[valid], iy[valid], ix[valid]]
        grad_norm = torch.norm(grad, dim=1, keepdim=True).clamp_min(1e-6)
        grad_dir = grad / grad_norm
        random_dir = torch.randn(grad_dir.shape, device=self.device, generator=self.rng)
        random_dir = random_dir / torch.norm(random_dir, dim=1, keepdim=True).clamp_min(1e-6)

        explore = self.drive_global[0] * drive_weights[:, 0] * self.graph.node_drive[:, 0]
        exploit = self.drive_global[1] * drive_weights[:, 1] * self.graph.node_drive[:, 1]
        wander = self.drive_global[2] * drive_weights[:, 2] * self.graph.node_drive[:, 2]

        agar = self.grid.grid[self.species_index["agar"], iy, ix]
        exploit = exploit * agar
        explore = explore * (1.0 - agar)

        impulse = (
            grad_dir * exploit.view(-1, 1)
            + random_dir * explore.view(-1, 1)
            + random_dir * wander.view(-1, 1)
        )
        self.graph.node_vel = self.graph.node_vel + impulse * mot_gain.view(-1, 1) * dt

    def apply_node_torque(self, dt, relax=0.08):
        if self.graph.edges.shape[0] == 0:
            return
        pos = self.graph.node_pos
        sum_dir = torch.zeros_like(pos)
        a = self.graph.edges[:, 0]
        b = self.graph.edges[:, 1]
        d = pos[b] - pos[a]
        norm = torch.norm(d, dim=1, keepdim=True).clamp_min(1e-6)
        dn = d / norm
        sum_dir = sum_dir.index_add(0, a, dn)
        sum_dir = sum_dir.index_add(0, b, -dn)
        sum_norm = torch.norm(sum_dir, dim=1, keepdim=True).clamp_min(1e-6)
        sum_dir = sum_dir / sum_norm
        rest = self.graph.node_rest_dir * (1.0 - relax) + sum_dir * relax
        rest = rest / torch.norm(rest, dim=1, keepdim=True).clamp_min(1e-6)
        self.graph.node_rest_dir = rest

        ra = rest[a]
        rb = rest[b]
        cross_a = ra[:, 0] * dn[:, 1] - ra[:, 1] * dn[:, 0]
        dot_a = ra[:, 0] * dn[:, 0] + ra[:, 1] * dn[:, 1]
        angle_a = torch.atan2(cross_a, dot_a)
        perp = torch.stack([-dn[:, 1], dn[:, 0]], dim=1)
        torque_a = perp * (-angle_a).unsqueeze(1)
        stiff = self.graph.node_stiffness
        self.graph.node_vel = self.graph.node_vel.index_add(0, a, torque_a * stiff[a].unsqueeze(1) * dt)
        self.graph.node_vel = self.graph.node_vel.index_add(0, b, -torque_a * stiff[a].unsqueeze(1) * dt)

        dn_b = -dn
        cross_b = rb[:, 0] * dn_b[:, 1] - rb[:, 1] * dn_b[:, 0]
        dot_b = rb[:, 0] * dn_b[:, 0] + rb[:, 1] * dn_b[:, 1]
        angle_b = torch.atan2(cross_b, dot_b)
        perp_b = torch.stack([-dn_b[:, 1], dn_b[:, 0]], dim=1)
        torque_b = perp_b * (-angle_b).unsqueeze(1)
        self.graph.node_vel = self.graph.node_vel.index_add(0, b, torque_b * stiff[b].unsqueeze(1) * dt)
        self.graph.node_vel = self.graph.node_vel.index_add(0, a, -torque_b * stiff[b].unsqueeze(1) * dt)

    def physics_step(self, dt):
        self.graph.node_age = self.graph.node_age + dt
        self.graph.node_phase = (self.graph.node_phase + dt / self.node_cycle_period) % 1.0
        if self.graph.edge_age.shape[0] > 0:
            self.graph.edge_age = self.graph.edge_age + dt
        prev_phase = self.global_phase
        self.global_phase = (self.global_phase + dt / self.global_period) % 1.0
        if prev_phase < self.bud_threshold <= self.global_phase:
            self.bud_ready = True

    def step(self, dt):
        self.physics_step(dt)
        self.update_node_metrics()
        self.apply_pellets(dt)
        self.apply_node_flux(dt)
        self.grid.step(dt)
        self.last_jacobian = compute_grid_jacobian(self.grid.grid)
        self.apply_motility(dt)
        self.apply_node_torque(dt)


def compute_grid_jacobian(grid):
    gx = 0.5 * (torch.roll(grid, shifts=-1, dims=2) - torch.roll(grid, shifts=1, dims=2))
    gy = 0.5 * (torch.roll(grid, shifts=-1, dims=1) - torch.roll(grid, shifts=1, dims=1))
    return torch.stack([gx, gy], dim=-1)


def demo():
    device = require_cuda()
    species = [
        Species("heat", 20.0, 0.02, diffusion=0.4),
        Species("nutrient", 120.0, 0.01, diffusion=0.35),
        Species("moisture", 210.0, 0.01, diffusion=0.35),
        Species("signal", 300.0, 0.02, diffusion=0.25, metric_enabled=True, metric_decay=0.05),
        Species("agar", 70.0, 0.005, diffusion=0.1),
    ]
    grid = VoxelGrid(64, 40, species, device)
    grid.grid[4].fill_(0.6)
    graph = Graph(len(species), device)
    a = graph.add_node((10, 10), initial=[0.1, 0.0, 0.0, 0.0, 0.3], kind="slime")
    b = graph.add_node((22, 14), initial=[0.0, 0.1, 0.0, 0.0, 0.2], kind="slime")
    graph.add_edge(a, b)
    node_types = {"slime": Slime()}
    world = World(grid, graph, node_types, device)
    world.add_pellet((8, 6), "agar", 0.05, radius=2.0)
    world.add_pellet((24, 16), "agar", 0.04, radius=2.0)
    run_visualizer(world, species)


def run_visualizer(world, species_defs):
    pygame.init()
    pygame.display.set_caption("Flux World CUDA")
    grid_w = world.grid.width
    grid_h = world.grid.height
    cell = 16
    margin = 24
    screen_w = grid_w * cell + margin * 2
    screen_h = grid_h * cell + margin * 2
    screen = pygame.display.set_mode((screen_w, screen_h))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    hues = torch.tensor([s.hue_deg for s in species_defs], device=world.device)
    base_cell = (170, 170, 165)
    edge_base = (205, 205, 205)
    node_base = (220, 220, 220)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        world.step(0.1)

        grid_cpu = world.grid.grid.detach().cpu()
        screen.fill((210, 210, 200))
        for y in range(grid_h):
            for x in range(grid_w):
                conc = grid_cpu[:, y, x]
                rgb = hue_blend(conc, hues.detach().cpu(), gain=1.0, base_rgb=base_cell)
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

        pos = world.graph.node_pos.detach().cpu()
        conc = world.graph.node_conc.detach().cpu()
        edges = world.graph.edges.detach().cpu()
        line_width = 5
        knob_radius = line_width // 2 + 1
        for edge in edges:
            a = int(edge[0].item())
            b = int(edge[1].item())
            pa = pos[a]
            pb = pos[b]
            ca = conc[a]
            cb = conc[b]
            rgb = hue_blend((ca + cb) * 0.5, hues.detach().cpu(), gain=1.0, base_rgb=edge_base)
            pygame.draw.line(
                screen,
                rgb,
                (int(margin + pa[0] * cell), int(margin + pa[1] * cell)),
                (int(margin + pb[0] * cell), int(margin + pb[1] * cell)),
                line_width,
            )
        for i in range(pos.shape[0]):
            rgb = hue_blend(conc[i], hues.detach().cpu(), gain=1.0, base_rgb=node_base)
            pygame.draw.circle(
                screen,
                rgb,
                (int(margin + pos[i, 0] * cell), int(margin + pos[i, 1] * cell)),
                knob_radius,
            )

        hud = f"nodes:{pos.shape[0]} edges:{edges.shape[0]}"
        text = font.render(hud, True, (20, 20, 20))
        screen.blit(text, (margin, 6))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    demo()
