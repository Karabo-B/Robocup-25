import numpy as np

def euclidean_distance(p1, p2):
    dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return dist

def role_assignment(teammate_positions, formation_positions): 
    n_players = len(teammate_positions)
    players_preferences = {}
    roles_preferences = {}
    point_preferences = {}
    

    for player_idx in range(n_players):
        player_pos = teammate_positions[player_idx]
        dist_to_roles = []
        
        for role_idx in range(n_players):
            role_pos = formation_positions[role_idx]
            distance = euclidean_distance(player_pos, role_pos)
            dist_to_roles.append((role_idx, distance))
        
        dist_to_roles.sort(key=lambda x: x[1])
        players_preferences[player_idx] = [role_idx for role_idx, dist in dist_to_roles]
        
    for role_idx in range(n_players):
        role_pos = formation_positions[role_idx]
        dist_to_players = []
        
        for player_idx in range(n_players):
            player_pos = teammate_positions[player_idx]
            distance = euclidean_distance(role_pos, player_pos)
            dist_to_players.append((player_idx, distance))
        
        dist_to_players.sort(key=lambda x: x[1])
        roles_preferences[role_idx] = [player_idx for player_idx, dist in dist_to_players]
        
    unmarried_players = list(range(n_players))
    role_assignments = {role: None for role in range(n_players)}
    marriage_proposals = {player: 0 for player in range(n_players)} 
    
    while unmarried_players:
        curr_player = unmarried_players[0]
        
        if marriage_proposals[curr_player] >= n_players:
            break
            
        target_role = players_preferences[curr_player][marriage_proposals[curr_player]]
        marriage_proposals[curr_player] += 1
        
        curr_role_occupant = role_assignments[target_role] 
        
        if curr_role_occupant is None:
            role_assignments[target_role] = curr_player
            unmarried_players.remove(curr_player)
        else:
            role_pref_list = roles_preferences[target_role]
            if role_pref_list.index(curr_player) < role_pref_list.index(curr_role_occupant):
                role_assignments[target_role] = curr_player
                unmarried_players.remove(curr_player)
                unmarried_players.append(curr_role_occupant)
    
    for role, player in role_assignments.items():
        point_preferences[player + 1] = formation_positions[role]


    return point_preferences