import math
import numpy as np
from math_ops.Math_Ops import Math_Ops as M

class Strategy:
    def __init__(self, world):
        self.mode = world.play_mode
        self.robot = world.robot
        self.my_head = self.robot.loc_head_position[:2]
        self.my_number = self.robot.unum
        self.my_position = np.array(world.teammates[self.my_number-1].state_abs_pos[:2])


        self.side = 1 if not world.team_side_is_left else 0

        self.teammates_pos = [
            tm.state_abs_pos[:2] if tm.state_abs_pos is not None else np.array([0.0, 0.0])
            for tm in world.teammates
        ]
        self.opponents_pos = [
            op.state_abs_pos[:2] if op.state_abs_pos is not None else np.array([0.0, 0.0])
            for op in world.opponents
        ]


        self.my_orientation = self.robot.imu_torso_orientation
        self.ball_pos = world.ball_abs_pos[:2]
        self.ball_vector = self.ball_pos - self.my_head
        self.ball_angle = M.vector_angle(self.ball_vector)
        self.ball_distance = np.linalg.norm(self.ball_vector)
        self.ball_sq_distance = self.ball_distance ** 2
        self.ball_speed = np.linalg.norm(world.get_ball_abs_vel(6)[:2])
        self.goal_angle = M.target_abs_angle(self.ball_pos, (15.05, 0))

        self.play_mode_group = world.play_mode_group
        self.slow_ball_pos = world.get_predicted_ball_pos(0.5)


        self.teammates_to_ball_sq = [
            np.sum((tm.state_abs_pos[:2] - self.slow_ball_pos) ** 2)
            if tm.state_last_update != 0 and (world.time_local_ms - tm.state_last_update <= 360 or tm.is_self) and not tm.state_fallen
            else 1000
            for tm in world.teammates
        ]
        self.opponents_to_ball_sq = [
            np.sum((op.state_abs_pos[:2] - self.slow_ball_pos) ** 2)
            if op.state_last_update != 0 and (world.time_local_ms - op.state_last_update <= 360) and not op.state_fallen
            else 1000
            for op in world.opponents
        ]


        self.closest_teammate_ball_sq = min(self.teammates_to_ball_sq)
        self.closest_teammate_ball_dist = math.sqrt(self.closest_teammate_ball_sq)
        self.closest_opponent_ball_dist = math.sqrt(min(self.opponents_to_ball_sq))
        self.active_player_number = self.teammates_to_ball_sq.index(self.closest_teammate_ball_sq) + 1

        
        self.target_pos, self.target_orientation = self._choose_role_position() #role position


    def _choose_role_position(self):
        ball_x, _ = self.ball_pos
        formation_number = self.my_number

  
        defensive = [(-13, 0), (-9, -4), (-3, 3), (3, 2), (7, 0)]
        neutral   = [(-8, 0), (-4, -3), (1, 2), (6, 2), (10, 0)]
        offensive = [(-5, 0), (0, -2), (5, 3), (9, 2), (12, 0)]

        if ball_x < -5:
            formation = defensive
        elif ball_x < 5:
            formation = neutral
        else:
            formation = offensive

        position = np.array(formation[formation_number - 1])

   
        if self.my_number == self.active_player_number:
            position = self.ball_pos

        angle = M.target_abs_angle(self.my_position, self.ball_pos)
        return position, angle


    def is_formation_ready(self, preferred_positions):
        ready = True
        for i in range(1, 6):
            if i != self.active_player_number:
                teammate_pos = self.teammates_pos[i - 1]
                if teammate_pos is not None:
                    dist_sq = np.sum((teammate_pos - preferred_positions[i]) ** 2)
                    if dist_sq > 0.3:
                        ready = False
        return ready

    def direction_to_target(self, target):
        vec = target - self.my_head
        return M.vector_angle(vec)

    def get_action(self):
        ball_x, ball_y = self.ball_pos
        my_num = self.my_number


        if my_num == self.active_player_number:
            if ball_x > 9:  #close to goal
                return 'kick', np.array([15.0, 0.0]), self.goal_angle


            my_x = self.teammates_pos[my_num - 1][0]
            ahead_teammates = [
                (i, pos) for i, pos in enumerate(self.teammates_pos)
                if pos[0] > my_x + 0.5
            ]

            if ahead_teammates:
                best_idx, best_score = None, -9999
                for i, pos in ahead_teammates:
                    to_goal = np.array([15.0, 0.0]) - pos
                    to_self = pos - self.ball_pos
                    alignment = np.dot(to_goal, to_self) / (np.linalg.norm(to_goal)*np.linalg.norm(to_self) + 1e-6)
                    dist = np.linalg.norm(pos - self.ball_pos)
                    score = alignment * 1.5 - dist * 0.3
                    if score > best_score:
                        best_score = score
                        best_idx = i
                target = self.teammates_pos[best_idx]
                return 'kick', target, self.goal_angle

            return 'move', np.array([15.0, 0.0]), self.goal_angle

        elif ball_x < 0:
            move_x = max(-13, ball_x - 2)
            move_y = np.clip(ball_y, -3, 3)
            return 'move', np.array([move_x, move_y]), self.direction_to_target(self.ball_pos)
        else:
            return 'move', self.target_pos, self.target_orientation
