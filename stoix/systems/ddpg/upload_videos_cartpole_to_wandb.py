import wandb
import imageio
import pickle
import numpy as np
import math

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
    def render(self, state, action_perf, action_safe, render_mode="human", extra_info = False):
        if action_perf == -10 or action_safe == [0,0] or action_safe == [-10,10]:
            return

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

        if action_safe!=[-10,-10] and extra_info:
            next_desired_action = action_perf
            a_h = action_safe[0]/abs(action_safe[0])
            b_h = action_safe[1]
            next_real_action = b_h/a_h if a_h * next_desired_action < b_h else next_desired_action
            next_threshold = b_h/a_h
            percentage_of_left_line = (next_threshold+1)/2.0
            percentage_of_real_action = (next_real_action+1)/2.0
            percentage_of_desired_action = (next_desired_action+1)/2.0
            to_right_is_dangerous = a_h < 0 # We project if a_h==1 and we are left of threshold
            left_color = (255,0,0) if not to_right_is_dangerous else (0,255,0)
            right_color = (255,0,0) if to_right_is_dangerous else (0,255,0)
            gfxdraw.hline(self.surf, int(self.screen_width / 4.0), int(self.screen_width / 4.0) + int ( (self.screen_width / 4.0 * 2.0)*percentage_of_left_line), 50, left_color)
            gfxdraw.hline(self.surf, int(self.screen_width / 4.0) + int ( (self.screen_width / 4.0 * 2.0)*percentage_of_left_line), int(self.screen_width / 4.0*3.0) , 50, right_color)
            gfxdraw.vline(self.surf,int(self.screen_width / 4.0)+ int ( (self.screen_width / 4.0 * 2.0)*percentage_of_real_action) ,40,60,(0,0,0))
            gfxdraw.vline(self.surf,int(self.screen_width / 4.0)+ int ( (self.screen_width / 4.0 * 2.0)*percentage_of_desired_action) ,45,55,(0,0,255))

        self.surf = pygame.transform.flip(self.surf, False, True)
        if action_safe!=[-10,-10] and extra_info:
            font = pygame.font.Font(None, 15)  # Default font, size 36
            text_surface = font.render("Drive Left (-1)", True, "black")
            text_rect = text_surface.get_rect(center=(int(self.screen_width / 4.0)-40, self.screen_height-(50)))
            self.surf.blit(text_surface, text_rect)
            text_surface = font.render("Drive Right (+1)", True, "black")
            text_rect = text_surface.get_rect(center=(int(self.screen_width / 4.0*3)+45, self.screen_height-(50)))
            self.surf.blit(text_surface, text_rect)
            text_surface = font.render("Desired Action", True, "black")
            text_rect = text_surface.get_rect(center=(int(self.screen_width / 4.0)+ int ( (self.screen_width / 4.0 * 2.0)*percentage_of_desired_action), self.screen_height-(30)))
            self.surf.blit(text_surface, text_rect)
            text_surface = font.render("Filtered Action", True, "black")
            text_rect = text_surface.get_rect(center=(int(self.screen_width / 4.0)+ int ( (self.screen_width / 4.0 * 2.0)*percentage_of_real_action), self.screen_height-(70)))
            self.surf.blit(text_surface, text_rect)
            font = pygame.font.Font(None, 25)
            text_surface = font.render(f"State: [{round(state[0],2),round(state[1],2),round(state[2],2),round(state[3],2)}]", True, "black")
            text_rect = text_surface.get_rect(center=(int(self.screen_width / 2.0), int(self.screen_height/4)))
            self.surf.blit(text_surface, text_rect)
            text_surface = font.render(f"Action_perf:{action_perf}, Action_safe:{action_safe}", True, "black")
            text_rect = text_surface.get_rect(center=(int(self.screen_width / 2.0), int(self.screen_height/4)-30))
            self.surf.blit(text_surface, text_rect)
        self.screen.blit(self.surf, (0, 0))
        if render_mode == "human":
            pygame.event.pump()
            self.clock.tick(50)
            pygame.display.flip()

        elif render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
def uploadVideos():
    #wandb.init(project="cleanRL", name = name)
    renderer = Renderer()
    with open("/home/artur/Schreibtisch/Stoix/experienced_trajectories.pkl", "rb") as file:
        loaded_list1 = pickle.load(file)
    with open("/home/artur/Schreibtisch/Stoix/experienced_actions_perf.pkl", "rb") as file:
        loaded_list2 = pickle.load(file)
    with open("/home/artur/Schreibtisch/Stoix/experienced_actions_safe.pkl", "rb") as file:
        loaded_list3 = pickle.load(file)
    for extra_info in [True, False]:
        for iteration1, iteration2, iteration3 in zip(loaded_list1, loaded_list2, loaded_list3):
            frames = []
            # The actions in it3 are not for the first reset state, they start with dummy action in gymnax.py reset
            iteration1= [[-10,-10,-10,-10]]+ iteration1
            iteration2 = [-10] + iteration2 
            iteration3 = iteration3 + [[-10,-10]]

            for state, action_perf, actions_safe in zip(iteration1, iteration2, iteration3):
                rgb_array = renderer.render(state,action_perf, actions_safe, render_mode="rgb_array", extra_info = extra_info)
                if not rgb_array is None:
                    frames.append(rgb_array)
            video_file = "trajectory.mp4"

            print(f"Video saved to {video_file}")
            if not extra_info:
                imageio.mimwrite(video_file, frames, fps=30)
                wandb.log({"simulation_video": wandb.Video(video_file, format="mp4")})
            else:
                imageio.mimwrite(video_file, frames, fps=2)
                wandb.log({"simulation_video_all_info": wandb.Video(video_file, format="mp4")})
