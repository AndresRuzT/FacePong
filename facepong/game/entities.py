"""Game entities: Paddles, Ball, Trails, and Particle System."""

from collections import deque
import math
import random
from typing import Deque, List, Optional, Tuple
import pygame

from facepong.config import PhysicsConfig


class Paddle:
    """Represents a player or AI controlled paddle with smooth position clamping."""

    def __init__(
        self,
        x: float,
        y: float,
        width: int,
        height: int,
        screen_height: int = 720,
        min_y: int = 10,
        max_y: Optional[int] = None,
        max_speed: float = 1250.0,
    ):
        self.x = float(x)
        self.y = float(y)  # Center Y coordinate
        self.width = width
        self.height = height
        self.screen_height = screen_height
        self.min_y = min_y
        self.max_y = max_y if max_y is not None else (screen_height - 10)

        self.target_y = float(y)
        self.speed = 650.0  # Pixels per second for keyboard or AI tracking
        self.max_speed = max_speed  # Pixels per second cap to eliminate teleportation jumps
        self.rect = pygame.Rect(0, 0, width, height)
        self._update_rect()

    def set_bounds(self, min_y: int, max_y: int) -> None:
        """Updates paddle vertical motion bounds."""
        self.min_y = min_y
        self.max_y = max_y
        self.clamp_target()

    def _update_rect(self) -> None:
        """Synchronizes Pygame bounding rectangle with center coordinates."""
        half_h = self.height / 2.0
        self.rect.x = int(self.x - self.width / 2.0)
        self.rect.y = int(self.y - half_h)

    def set_target_normalized_y(self, norm_y: float) -> None:
        """Maps a 0.0-1.0 normalized coordinate to valid screen paddle range."""
        half_h = self.height / 2.0
        min_center = self.min_y + half_h + 4
        max_center = self.max_y - half_h - 4
        self.target_y = min_center + norm_y * (max_center - min_center)

    def reset_to_center(self, center_y: float) -> None:
        """Immediately snaps paddle to vertical center coordinate."""
        self.target_y = float(center_y)
        self.y = float(center_y)
        self.clamp_target()
        self._update_rect()

    def move_keyboard(self, direction: float, dt: float) -> None:
        """Moves paddle by directional input (-1.0 up, 1.0 down)."""
        self.target_y += direction * self.speed * dt
        self.clamp_target()

    def update(self, dt: float, smooth_factor: float = 32.0) -> None:
        """
        Smoothly interpolates paddle position towards target coordinate,
        clamping step velocity to max_speed to eliminate teleportation jumps.
        """
        diff = self.target_y - self.y
        desired_velocity = diff * smooth_factor
        clamped_velocity = max(-self.max_speed, min(self.max_speed, desired_velocity))
        self.y += clamped_velocity * dt

        # Enforce bounds
        half_h = self.height / 2.0
        min_center = self.min_y + half_h + 4
        max_center = self.max_y - half_h - 4
        self.y = max(min_center, min(max_center, self.y))
        self._update_rect()

    def clamp_target(self) -> None:
        """Ensures target coordinate does not exceed boundaries."""
        half_h = self.height / 2.0
        min_center = self.min_y + half_h + 4
        max_center = self.max_y - half_h - 4
        self.target_y = max(min_center, min(max_center, self.target_y))


class Ball:
    """Ball entity featuring high precision trajectory, comet trail, and bounce physics."""

    def __init__(self, screen_width: int, screen_height: int, config: PhysicsConfig, arena_rect: Optional[pygame.Rect] = None):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.config = config
        self.radius = config.ball_radius

        self.arena_rect = arena_rect or pygame.Rect(0, 10, screen_width, screen_height - 20)
        self.x = float(self.arena_rect.centerx)
        self.y = float(self.arena_rect.centery)
        self.vx = 0.0
        self.vy = 0.0
        self.speed = config.ball_initial_speed

        self.trail: Deque[Tuple[float, float]] = deque(maxlen=10)
        self.trail_timer = 0.0

    def set_arena(self, arena_rect: pygame.Rect) -> None:
        """Updates active arena boundaries."""
        self.arena_rect = arena_rect

    def serve(self, direction_to_player: bool = True) -> None:
        """Resets the ball to arena center and launches with randomized angle."""
        self.x = float(self.arena_rect.centerx)
        self.y = float(self.arena_rect.centery)
        self.speed = self.config.ball_initial_speed
        self.trail.clear()

        # Launch angle between -35 and +35 degrees
        angle_deg = random.uniform(-35.0, 35.0)
        angle_rad = math.radians(angle_deg)

        dir_x = -1.0 if direction_to_player else 1.0
        self.vx = dir_x * self.speed * math.cos(angle_rad)
        self.vy = self.speed * math.sin(angle_rad)

    def update(self, dt: float) -> Tuple[bool, bool]:
        """
        Advances ball position and checks top/bottom wall collisions.
        
        Returns:
            Tuple of (bounced_wall, bounced_floor_or_ceiling)
        """
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Update trail history at regular intervals
        self.trail_timer += dt
        if self.trail_timer >= 0.016:
            self.trail.append((self.x, self.y))
            self.trail_timer = 0.0

        # Boundary bounce check
        wall_bounce = False
        top_limit = self.arena_rect.top + self.radius + 2
        bottom_limit = self.arena_rect.bottom - self.radius - 2

        if self.y <= top_limit:
            self.y = top_limit
            self.vy = abs(self.vy)
            wall_bounce = True
        elif self.y >= bottom_limit:
            self.y = bottom_limit
            self.vy = -abs(self.vy)
            wall_bounce = True

        return wall_bounce, False

    def handle_paddle_collision(self, paddle: Paddle, is_player: bool) -> bool:
        """
        Calculates deflection angle and speed increase when hitting a paddle.
        """
        # Quick bounding box proximity check
        ball_rect = pygame.Rect(
            int(self.x - self.radius),
            int(self.y - self.radius),
            self.radius * 2,
            self.radius * 2,
        )

        if not ball_rect.colliderect(paddle.rect):
            return False

        # Ensure ball is traveling towards the paddle
        if is_player and self.vx > 0:
            return False
        if not is_player and self.vx < 0:
            return False

        # Hit point relative to paddle center [-1.0, 1.0]
        offset = (self.y - paddle.y) / (paddle.height / 2.0)
        offset = max(-1.0, min(1.0, offset))

        # Reflection angle
        max_angle = math.radians(self.config.max_bounce_angle_deg)
        bounce_angle = offset * max_angle

        # Accelerate ball slightly per hit
        self.speed = min(self.config.ball_max_speed, self.speed + self.config.ball_speed_step)

        direction_x = 1.0 if is_player else -1.0
        self.vx = direction_x * self.speed * math.cos(bounce_angle)
        self.vy = self.speed * math.sin(bounce_angle)

        # Reposition ball outside paddle bounding box to prevent sticky bounce
        if is_player:
            self.x = paddle.rect.right + self.radius + 1
        else:
            self.x = paddle.rect.left - self.radius - 1

        return True


class Particle:
    """Individual particle for arcade collision sparks and goal fireworks."""

    def __init__(self, x: float, y: float, color: Tuple[int, int, int], speed_range: Tuple[float, float]):
        self.x = x
        self.y = y
        self.color = color

        angle = random.uniform(0.0, 2.0 * math.pi)
        speed = random.uniform(speed_range[0], speed_range[1])
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

        self.life = random.uniform(0.2, 0.45)
        self.max_life = self.life
        self.size = random.uniform(2.0, 4.5)

    def update(self, dt: float) -> bool:
        """Updates particle kinematics; returns False when life expires."""
        self.x += self.vx * dt
        self.y += self.vy * dt
        # Drag friction
        self.vx *= 0.94
        self.vy *= 0.94
        self.life -= dt
        return self.life > 0.0


class ParticleSystem:
    """Manages particle emitters with a strict budget for embedded devices."""

    def __init__(self, max_particles: int = 120):
        self.max_particles = max_particles
        self.particles: List[Particle] = []

    def emit(self, x: float, y: float, color: Tuple[int, int, int], count: int = 15, speed_range=(80.0, 280.0)) -> None:
        """Spawns an explosion burst of particles."""
        available_slots = max(0, self.max_particles - len(self.particles))
        actual_count = min(count, available_slots)
        for _ in range(actual_count):
            self.particles.append(Particle(x, y, color, speed_range))

    def update(self, dt: float) -> None:
        """Updates all living particles."""
        self.particles = [p for p in self.particles if p.update(dt)]
