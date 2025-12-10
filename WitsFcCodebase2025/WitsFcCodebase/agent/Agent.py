from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np

from strategy.Assignment import role_assignment 
from strategy.Strategy import Strategy 
from formation.Formation import GenerateBasicFormation


class Agent(Base_Agent):

    def __init__(self, host: str, agent_port: int, monitor_port: int, unum: int,
                 team_name: str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False):
        
        robot_kind = (0, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4)[unum-1]
        super().__init__(host, agent_port, monitor_port, unum, robot_kind, team_name,
                         enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3)

        self.init_pos = ([-14,0], [-9,-5], [-9,0], [-9,5], [-5,-5], [-5,0], [-5,5],
                         [-1,-6], [-1,-2.5], [-1,2.5], [-1,6])[unum-1]

    def beam(self, avoid_center_circle=False):
        robot = self.world.robot
        pos = self.init_pos[:]
        self.state = 0

        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3 

        if np.linalg.norm(pos - robot.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos, M.vector_angle((-pos[0], -pos[1])))
        else:
            if self.fat_proxy_cmd is None:
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else:
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3)


    def move(self, target=(0, 0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        robot = self.world.robot

        if self.fat_proxy_cmd is not None:
            self.fat_proxy_move(target, orientation, is_orientation_absolute)
            return

        if avoid_obstacles:
            target, _, distance_to_final = self.path_manager.get_path_to_target(
                target, priority_unums=priority_unums,
                is_aggressive=is_aggressive, timeout=timeout
            )
        else:
            distance_to_final = np.linalg.norm(target - robot.loc_head_position[:2])

        self.behavior.execute("Walk", target, True, orientation, is_orientation_absolute, distance_to_final)

    def kick(self, kick_direction=None, kick_distance=None,
             abort=False, enable_pass_command=False):
        return self.behavior.execute("Dribble", None, None)

    def kickTarget(self, strategy_data, my_pos=(0,0), target=(0,0),
               abort=False, enable_pass_command=False):
        
        vector = [target[0] - my_pos[0], target[1] - my_pos[1]] #from mypos to target
        distance = math.hypot(vector[0], vector[1])
        direction = math.degrees(math.atan2(vector[1], vector[0]))

        if strategy_data.closest_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()


        self.kick_direction = self.kick_direction if direction is None else direction
        self.kick_distance = self.kick_distance if distance is None else distance


        if self.fat_proxy_cmd is None:
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort)
        else:
            return self.fat_proxy_kick()


    def think_and_send(self):
        behavior = self.behavior
        strategy_data = Strategy(self.world)
        drawer = self.world.draw

        if strategy_data.mode == self.world.M_GAME_OVER:
            pass
        elif strategy_data.play_mode_group == self.world.MG_ACTIVE_BEAM:
            self.beam()
        elif strategy_data.play_mode_group == self.world.MG_PASSIVE_BEAM:
            self.beam(True)
        elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            self.state = 0 if behavior.execute("Get_Up") else 1
        else:
            if strategy_data.mode != self.world.M_BEFORE_KICKOFF:
                self.select_skill(strategy_data)

        self.radio.broadcast()

        if self.fat_proxy_cmd is None:
            self.scom.commit_and_send(strategy_data.robot.get_command())
        else:
            self.scom.commit_and_send(self.fat_proxy_cmd.encode())
            self.fat_proxy_cmd = ""


    def select_skill(self, strategy_data):
        drawer = self.world.draw


        if strategy_data.active_player_number == strategy_data.robot.unum:
            drawer.annotation((0, 10.5), "Role Assignment Phase", drawer.Color.yellow, "status")
        else:
            drawer.clear("status")

        formation_positions = GenerateBasicFormation()
        point_preferences = role_assignment(strategy_data.teammates_pos, formation_positions)
        strategy_data.target_pos = point_preferences[strategy_data.my_number]
        strategy_data.target_orientation = strategy_data.direction_to_target(strategy_data.target_pos)

        drawer.line(strategy_data.my_position, strategy_data.target_pos, 2, drawer.Color.blue, "target line")


        if strategy_data.active_player_number == strategy_data.robot.unum:
            drawer.annotation((0, 10.5), "Decision Phase", drawer.Color.yellow, "status")

            # Decide action: move or kick
            action, target, orientation = strategy_data.get_action()

            drawer.line(strategy_data.my_position, target, 2,
                        drawer.Color.red if action == 'kick' else drawer.Color.green,
                        "pass line")

            if action == "kick":
                return self.kickTarget(strategy_data, strategy_data.my_position, target)
            else:
                return self.move(target, orientation=orientation)
        else:
            drawer.clear("pass line")
            return self.move(strategy_data.target_pos, orientation=strategy_data.ball_angle)
