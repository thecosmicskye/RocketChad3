import torch
import rocketgpu_py as rs

class GoalTelemetry:

    def __init__(self, sim, device, capacity):
        layout = rs.physics_event_layout()
        self.layout = layout
        n = sim.env_count() if callable(sim.env_count) else sim.env_count
        stride = layout['record_size']
        self.records = torch.zeros((n, capacity, stride), dtype=torch.uint8, device=device)
        self.windows = torch.zeros((n, layout['window_size'] // 4), dtype=torch.int32, device=device)
        self.types = self.records[:, :, layout['type_offset']]
        self.velocities = self.records.view(torch.float32)[:, :, layout['velocity_offset'] // 4:layout['velocity_offset'] // 4 + 3]
        self.counts = self.windows[:, layout['count_offset'] // 4]
        self.overflow = self.windows[:, layout['overflow_offset'] // 4]
        self.indices = torch.arange(capacity, device=device)[None]
        self.envs = torch.arange(n, device=device)
        self.sim = sim
        sim.set_physics_event_stream(self.records.data_ptr(), self.windows.data_ptr(), capacity, goal_only=True)

    def begin(self):
        self.counts.zero_()
        self.overflow.zero_()

    def read(self, scored):
        mask = (self.indices < self.counts[:, None]) & (self.types == self.layout['goal_type'])
        first = mask.to(torch.int32).argmax(1)
        found = mask.any(1)
        speed = torch.linalg.vector_norm(self.velocities[self.envs, first], dim=-1)
        speed = torch.where(found, speed, 0.0)
        invalid = (self.overflow != 0) | (found != scored)
        return dict(goal_event_count=found.sum(), goal_speed_sum=speed.sum(), goal_speed_squared_sum=speed.square().sum(), goal_event_invalid=invalid.sum(), goal_found=found, goal_speed=speed)
