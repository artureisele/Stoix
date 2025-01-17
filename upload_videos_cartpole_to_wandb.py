import wandb
import imageio
import pickle
import numpy as np

class Renderer():
    def __init__(self):
        self.screen = None
        self.screen_width = 704
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.gravity: float = 9.8
        self.masscart: float = 1.0
        self.masspole: float = 0.1
        self.total_mass: float = 1.0 + 0.1  # (masscart + masspole)
        self.length: float = 0.5
        self.polemass_length: float = 0.05  # (masspole * length)
        self.force_mag: float = 30.0
        self.tau: float = 0.02
        self.theta_threshold_radians: float = 24 * 2 * np.pi / 360
        self.x_threshold: float = 2.4
        self.max_steps_in_episode: int = 500  # v0 had only 200 steps!
    def render(self, state, render_mode="human"):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
                print('pygame is not installed, run `pip install "gymnasium[classic-control]"`')
        if self.screen is None:
            pygame.init()
            if render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2 +1.2
        scale = self.screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0
        if state is None:
            return None
        x0 = state[0]
        x2 = state[2]

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x0 * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x2)
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))
        gfxdraw.vline(self.surf,int(self.x_threshold * scale + self.screen_width / 2.0), 0, self.screen_height, (255, 0, 0))
        gfxdraw.vline(self.surf,int(-self.x_threshold * scale + self.screen_width / 2.0),0,self.screen_height, (255, 0, 0))
        gfxdraw.vline(self.surf,int(1.7 * scale + self.screen_width / 2.0),0,self.screen_height, (0, 255, 255))
        """
        if self.debug_hyperplanes_render:
            next_desired_action = self.next_desired_action
            next_real_action = self.next_real_action
            next_threshold = self.next_threshold
            percentage_of_left_line = (next_threshold+1)/2.0
            percentage_of_real_action = (next_real_action+1)/2.0
            percentage_of_desired_action = (next_desired_action+1)/2.0
            to_right_is_dangerous = self.to_right_is_dangerous
            left_color = (255,0,0) if not to_right_is_dangerous else (0,255,0)
            right_color = (255,0,0) if to_right_is_dangerous else (0,255,0)
            gfxdraw.hline(self.surf, int(self.screen_width / 4.0), int(self.screen_width / 4.0) + int ( (self.screen_width / 4.0 * 2.0)*percentage_of_left_line), 50, left_color)
            gfxdraw.hline(self.surf, int(self.screen_width / 4.0) + int ( (self.screen_width / 4.0 * 2.0)*percentage_of_left_line), int(self.screen_width / 4.0*3.0) , 50, right_color)
            gfxdraw.vline(self.surf,int(self.screen_width / 4.0)+ int ( (self.screen_width / 4.0 * 2.0)*percentage_of_real_action) ,25,75,(0,0,0))
            gfxdraw.vline(self.surf,int(self.screen_width / 4.0)+ int ( (self.screen_width / 4.0 * 2.0)*percentage_of_desired_action) ,35,65,(0,0,255))
        """
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if render_mode == "human":
            pygame.event.pump()
            self.clock.tick(50)
            pygame.display.flip()

        elif render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
if __name__ == "__main__":
    wandb.init(project="cleanRL")
    renderer = Renderer()
    with open("experienced_trajectories.pkl", "rb") as file:
        loaded_list = pickle.load(file)
    for iteration in loaded_list:
        frames = []
        for state in iteration:
            rgb_array = renderer.render(state, render_mode="rgb_array")
            frames.append(rgb_array)
        video_file = "trajectory.mp4"
        imageio.mimwrite(video_file, frames, fps=30)
        print(f"Video saved to {video_file}")
        wandb.log({"simulation_video": wandb.Video(video_file, fps=30, format="mp4")})
