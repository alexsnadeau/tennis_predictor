BASE_ELO = 1500

def createStats(base_elo=BASE_ELO):
    from collections import defaultdict, deque

    prev_stats = {}
    prev_stats["elo_players"] = defaultdict(lambda: base_elo)
    prev_stats["elo_surface_players"] = defaultdict(lambda: defaultdict(lambda: base_elo))
    prev_stats["elo_grad_players"] = defaultdict(lambda: deque(maxlen=1000))
    prev_stats["last_k_matches"] = defaultdict(lambda: deque(maxlen=1000))
    prev_stats["last_k_matches_stats"] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=1000)))
    prev_stats["matches_played"] = defaultdict(int)
    prev_stats["matches_surface_played"] = defaultdict(lambda: defaultdict(int))
    prev_stats["h2h"] = defaultdict(int)
    prev_stats["h2h_surface"] = defaultdict(lambda: defaultdict(int))
    prev_stats["last_tourney"] = defaultdict(lambda: deque(maxlen=20))
    prev_stats["last_tourney_surface"] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=20)))

    return prev_stats

from datetime import datetime
from typing import Sequence, List

def days_between(date1: str, date2: str) -> int:
    d1 = datetime.strptime(date1, "%Y%m%d")
    d2 = datetime.strptime(date2, "%Y%m%d")
    return abs((d2 - d1).days)

def gaps_between_tourneys(played: Sequence[str]) -> List[int]:
    """
    Day gaps between consecutive YYYYMMDD strings, most recent first.
    """
    if len(played) < 2:
        return []
    sorted_dates = sorted(played)  # YYYYMMDD sorts chronologically
    gaps = [days_between(sorted_dates[i-1], sorted_dates[i])
            for i in range(1, len(sorted_dates))]
    gaps.reverse()  # most recent gap first
    return gaps

def average_days_between_tourneys(played: Sequence[str]) -> float:
    """
    Average day gap; 0.0 if fewer than two dates.
    """
    gaps = gaps_between_tourneys(played)
    return (sum(gaps) / len(gaps)) if gaps else 0.0

def k_bonus_after_layoff(played: Sequence[str], threshold: int = 100) -> float:
    """
    Start at 1.5x if the most recent gap is a 'layoff' (>= threshold).
    For each consecutive recent short gap (< threshold), reduce by 0.05.
    Require enough history (>=16 gaps) to apply; otherwise return 1.0.
    Never go below 1.0.
    """
    gaps = gaps_between_tourneys(played)
    if len(gaps) <= 15:
        return 1.0

    bonus = 1.5
    for g in gaps:          # most recent first
        if g < threshold:
            bonus -= 0.05
        else:
            break
    return max(1.0, bonus)

def calculate_k_factor(n_games: int,
                       base_k: float,
                       max_k: float,
                       div_number: float,
                       last_played: Sequence[str],
                       bonus_after_layoff: bool) -> float:
    """
    Combine base/div component with layoff bonus, capped at max_k.
    """
    if bonus_after_layoff:
        bonus_mul = k_bonus_after_layoff(last_played)
    else:
        bonus_mul = 1
    
    k = (base_k + div_number / (n_games + 1)) * bonus_mul
    return min(k, max_k)

def round_importance(code: str) -> int:
    ROUND_RANK = {
        'F': 1,
        'BR': 2,
        'SF': 3,
        'QF': 4,
        'R16': 5,
        'R32': 6,
        'R64': 7,
        'R128': 8,
        'RR': 9,
        'ER': 10,
        'Q3': 11,
        'Q2': 12,
        'Q1': 13,
    }
    return ROUND_RANK[code]

"""
INPUTS:
Match should be a row in the tennis dataset we created in 0.CleanData.ipynb
Prev_stats should be all the stats data we have until the most recent game. We want to update these stats and return a dictionary again

OUTPUT:
Outputs a dictionary with the updated stats
"""
def updateStats(match, prev_stats, k_factor, base_k_factor, max_k_factor, div_number, bonus_after_layoff):
    from utils.common import mean, getWinnerLoserIDS
    import numpy as np

    # Get Winner and Loser ID'S
    p1_id, p2_id, surface, result = match.p1_id, match.p2_id, match.surface, match.RESULT
    w_id, l_id = getWinnerLoserIDS(p1_id, p2_id, result)

    ######################## UPDATE ########################
    ########################################################
    ########### Last Tourney Date and on surface ###########
    prev_stats["last_tourney"][w_id].append(str(match.tourney_date))
    prev_stats["last_tourney"][l_id].append(str(match.tourney_date))
    prev_stats["last_tourney_surface"][surface][w_id].append(str(match.tourney_date))
    prev_stats["last_tourney_surface"][surface][l_id].append(str(match.tourney_date))
    
    ############## ELO ##############
    # Decide initial elo in case player doesn't have one
    initial_elo = BASE_ELO
    
    # Get current ELO ratings (BEFORE this match)
    elo_w = prev_stats["elo_players"].get(w_id, initial_elo)
    elo_l = prev_stats["elo_players"].get(l_id, initial_elo)
    elo_surface_w = prev_stats["elo_surface_players"][surface].get(w_id, initial_elo)
    elo_surface_l = prev_stats["elo_surface_players"][surface].get(l_id, initial_elo)

    # Calculate expected probabilities
    if k_factor is not None:
        k_factor_w = k_factor_l = k_factor_surface_w = k_factor_surface_l = k_factor
    else:
        # instead of .get(..., match.tourney_date)
        w_seq_all = prev_stats["last_tourney"][w_id]
        l_seq_all = prev_stats["last_tourney"][l_id]
        w_seq_srf = prev_stats["last_tourney_surface"][surface][w_id]
        l_seq_srf = prev_stats["last_tourney_surface"][surface][l_id]

        k_factor_w = calculate_k_factor(prev_stats["matches_played"][w_id], base_k_factor, max_k_factor, div_number, w_seq_all, bonus_after_layoff)
        k_factor_l = calculate_k_factor(prev_stats["matches_played"][l_id], base_k_factor, max_k_factor, div_number, l_seq_all, bonus_after_layoff)
        k_factor_surface_w = calculate_k_factor(prev_stats["matches_surface_played"][surface][w_id], base_k_factor, max_k_factor, div_number, w_seq_srf, bonus_after_layoff)
        k_factor_surface_l = calculate_k_factor(prev_stats["matches_surface_played"][surface][l_id], base_k_factor, max_k_factor, div_number, l_seq_srf, bonus_after_layoff)

    
    exp_w = 1/(1+(10**((elo_l-elo_w)/400)))
    exp_l = 1/(1+(10**((elo_w-elo_l)/400)))
    exp_surface_w = 1/(1+(10**((elo_surface_l-elo_surface_w)/400)))
    exp_surface_l = 1/(1+(10**((elo_surface_w-elo_surface_l)/400)))

    # Update ELO ratings for next match
    elo_w += k_factor_w*(1-exp_w)
    elo_l += k_factor_l*(0-exp_l)
    elo_surface_w += k_factor_surface_w*(1-exp_surface_w)
    elo_surface_l += k_factor_surface_l*(0-exp_surface_l)

    # Store updated ratings
    prev_stats["elo_players"][w_id] = elo_w
    prev_stats["elo_players"][l_id] = elo_l
    prev_stats["elo_surface_players"][surface][w_id] = elo_surface_w
    prev_stats["elo_surface_players"][surface][l_id] = elo_surface_l

    ########################################################
    ############## ELO GRAD ##############
    prev_stats["elo_grad_players"][w_id].append(elo_w)
    prev_stats["elo_grad_players"][l_id].append(elo_l)
    
    ########################################################
    ############## Matches Played ##############
    prev_stats["matches_played"][w_id] += 1
    prev_stats["matches_played"][l_id] += 1
    prev_stats["matches_surface_played"][surface][w_id] += 1
    prev_stats["matches_surface_played"][surface][l_id] += 1

    ########################################################
    ############## % Last K Matches Won ##############
    prev_stats["last_k_matches"][w_id].append(1)
    prev_stats["last_k_matches"][l_id].append(0)

    ########################################################
    ############# H2H and H2H on that surface #############
    prev_stats["h2h"][(w_id, l_id)] += 1
    prev_stats["h2h_surface"][surface][(w_id, l_id)] += 1

    ########################################################
    ############# UPDATE Various Ohter Statistics #############
    if p1_id == getWinnerLoserIDS(p1_id, p2_id, result)[0]:
        w_ace, l_ace = match.p1_ace, match.p2_ace
        w_df, l_df = match.p1_df, match.p2_df
        w_svpt, l_svpt = match.p1_svpt, match.p2_svpt
        w_1stIn, l_1stIn = match.p1_1stIn, match.p2_1stIn
        w_1stWon, l_1stWon = match.p1_1stWon, match.p2_1stWon
        w_2ndWon, l_2ndWon = match.p1_2ndWon, match.p2_2ndWon
        w_bpSaved, l_bpSaved = match.p1_bpSaved, match.p2_bpSaved
        w_bpFaced, l_bpFaced = match.p1_bpFaced, match.p2_bpFaced
    else:
        w_ace, l_ace = match.p2_ace, match.p1_ace
        w_df, l_df = match.p2_df, match.p1_df
        w_svpt, l_svpt = match.p2_svpt, match.p1_svpt
        w_1stIn, l_1stIn = match.p2_1stIn, match.p1_1stIn
        w_1stWon, l_1stWon = match.p2_1stWon, match.p1_1stWon
        w_2ndWon, l_2ndWon = match.p2_2ndWon, match.p1_2ndWon
        w_bpSaved, l_bpSaved = match.p2_bpSaved, match.p1_bpSaved
        w_bpFaced, l_bpFaced = match.p2_bpFaced, match.p1_bpFaced
    
    #######################
    ##### Serve Stats #####
    if (w_svpt != 0) and (w_svpt != w_1stIn):
        # Percentage of aces
        prev_stats["last_k_matches_stats"][w_id]["p_ace"].append(100*(w_ace/w_svpt))
        # Percentage of double faults
        prev_stats["last_k_matches_stats"][w_id]["p_df"].append(100*(w_df/w_svpt))
        # Percentage of first serve in
        prev_stats["last_k_matches_stats"][w_id]["p_1stIn"].append(100*(w_1stIn/w_svpt))
        # Percentage of second serve won
        prev_stats["last_k_matches_stats"][w_id]["p_2ndWon"].append(100*(w_2ndWon/(w_svpt-w_1stIn)))
    if l_svpt != 0 and (l_svpt != l_1stIn):
        prev_stats["last_k_matches_stats"][l_id]["p_ace"].append(100*(l_ace/l_svpt))
        prev_stats["last_k_matches_stats"][l_id]["p_df"].append(100*(l_df/l_svpt))
        prev_stats["last_k_matches_stats"][l_id]["p_1stIn"].append(100*(l_1stIn/l_svpt))
        prev_stats["last_k_matches_stats"][l_id]["p_2ndWon"].append(100*(l_2ndWon/(l_svpt-l_1stIn)))

    # Percentage of first serve won
    if w_1stIn != 0:
        prev_stats["last_k_matches_stats"][w_id]["p_1stWon"].append(100*(w_1stWon/w_1stIn))
    if l_1stIn != 0:
        prev_stats["last_k_matches_stats"][l_id]["p_1stWon"].append(100*(l_1stWon/l_1stIn))
    
    # Percentage of second serve won
    if w_bpFaced != 0:
        prev_stats["last_k_matches_stats"][w_id]["p_bpSaved"].append(100*(w_bpSaved/w_bpFaced))
    if l_bpFaced != 0:
        prev_stats["last_k_matches_stats"][l_id]["p_bpSaved"].append(100*(l_bpSaved/l_bpFaced))
    
    ########################
    ##### Return Stats #####
    # Winner return vs. loser serve
    if l_svpt != 0:
        # Overall return points won
        w_rpw = (l_svpt - l_1stWon - l_2ndWon) / l_svpt
        prev_stats["last_k_matches_stats"][w_id]["p_rpw"].append(100*(w_rpw))
        # Aces against while returning (opponent aces per opponent serve point)
        prev_stats["last_k_matches_stats"][w_id]["p_retAceAgainst"].append(100*(l_ace / l_svpt))
    if l_1stIn != 0:
        prev_stats["last_k_matches_stats"][w_id]["p_ret1stWon"].append(100*((l_1stIn - l_1stWon) / l_1stIn))
    if (l_svpt - l_1stIn) != 0:
        prev_stats["last_k_matches_stats"][w_id]["p_ret2ndWon"].append(100*(((l_svpt - l_1stIn) - l_2ndWon) / (l_svpt - l_1stIn)))
    # Break-point conversion (on return))
    if l_bpFaced != 0:
        prev_stats["last_k_matches_stats"][w_id]["p_bpConv"].append(100*((l_bpFaced - l_bpSaved) / l_bpFaced))

    # Loser return vs. winner serve
    if w_svpt != 0:
        l_rpw = (w_svpt - w_1stWon - w_2ndWon) / w_svpt
        prev_stats["last_k_matches_stats"][l_id]["p_rpw"].append(100*(l_rpw))
        prev_stats["last_k_matches_stats"][l_id]["p_retAceAgainst"].append(100*(w_ace / w_svpt))
    if w_1stIn != 0:
        prev_stats["last_k_matches_stats"][l_id]["p_ret1stWon"].append(100*((w_1stIn - w_1stWon) / w_1stIn))
    if (w_svpt - w_1stIn) != 0:
        prev_stats["last_k_matches_stats"][l_id]["p_ret2ndWon"].append(100*(((w_svpt - w_1stIn) - w_2ndWon) / (w_svpt - w_1stIn)))
    if w_bpFaced != 0:
        prev_stats["last_k_matches_stats"][l_id]["p_bpConv"].append(100*((w_bpFaced - w_bpSaved) / w_bpFaced))

    ############################
    ##### Total Points Won #####
    total_pts = w_svpt + l_svpt
    if total_pts != 0:
        # Winner TPW%
        w_tpw = (w_1stWon + w_2ndWon) + (l_svpt - l_1stWon - l_2ndWon)
        prev_stats["last_k_matches_stats"][w_id]["p_totalPtsWon"].append(100*(w_tpw / total_pts))
        # Loser TPW%
        l_tpw = (l_1stWon + l_2ndWon) + (w_svpt - w_1stWon - w_2ndWon)
        prev_stats["last_k_matches_stats"][l_id]["p_totalPtsWon"].append(100*(l_tpw / total_pts))

    ##########################
    ##### Dominance Ratio ####
    # DR = (Return Points Won %) / (Serve Points Lost %)
    # Serve Points Lost % = 1 - (SPW / SVPT)
    if (w_svpt != 0) and (l_svpt != 0):
        w_spw = (w_1stWon + w_2ndWon) / w_svpt
        l_spw = (l_1stWon + l_2ndWon) / l_svpt
        w_spl = 1.0 - w_spw
        l_spl = 1.0 - l_spw
        # Append only if denominator > 0 to avoid division by zero
        if w_spl > 0:
            prev_stats["last_k_matches_stats"][w_id]["dominance_ratio"].append(100*(w_rpw / w_spl))
        if l_spl > 0:
            prev_stats["last_k_matches_stats"][l_id]["dominance_ratio"].append(100*(l_rpw / l_spl))
    
    return prev_stats

"""
INPUTS:
Player1 and Player2 should be a dictionaries with the following keys for each player: ID, ATP_POINTS, ATP_RANK, AGE, HEIGHT, 
Match should be a dict with common information about the game (like the number of BEST_OF, DRAW_SIZE, SURFACE). (cool thing is that in the future we can add more stuff here).
Prev_stats should be all the stats data we have until the most recent game. If we were predicting new data, we would just pass all the calculated stats in the dataset from 1991 to now.

OUTPUT:
Outputs a dictionary with all the stats calcualted
"""
def getStats(player1, player2, match, prev_stats):
    from utils.common import mean, getWinnerLoserIDS
    import numpy as np

    output = {}
    PLAYER1_ID = player1["ID"]
    PLAYER2_ID = player2["ID"]
    SURFACE = match["SURFACE"]

    # Get Differences
    output["BEST_OF"] = match["BEST_OF"]
    output["DRAW_SIZE"] = match["DRAW_SIZE"]
    output["ROUND"] = round_importance(match["ROUND"])
    output["AGE_DIFF"] = player1["AGE"]-player2["AGE"]
    output["HEIGHT_DIFF"] = player1["HEIGHT"]-player2["HEIGHT"]
    output["ATP_RANK_DIFF"] = player1["ATP_RANK"]-player2["ATP_RANK"]

    # Get Stats from Dictionary
    elo_players = prev_stats["elo_players"]
    elo_surface_players = prev_stats["elo_surface_players"]
    elo_grad_players = prev_stats["elo_grad_players"]
    last_k_matches = prev_stats["last_k_matches"]
    last_k_matches_stats = prev_stats["last_k_matches_stats"]
    matches_played = prev_stats["matches_played"]
    h2h = prev_stats["h2h"]
    h2h_surface = prev_stats["h2h_surface"]

    ####################### GET STATS ########################
    output["ELO_DIFF"] = elo_players[PLAYER1_ID] - elo_players[PLAYER2_ID]
    output["ELO_SURFACE_DIFF"] = elo_surface_players[SURFACE][PLAYER1_ID] - elo_surface_players[SURFACE][PLAYER2_ID]
    output["N_GAMES_DIFF"] = matches_played[PLAYER1_ID] - matches_played[PLAYER2_ID]
    output["H2H_DIFF"] = h2h[(PLAYER1_ID, PLAYER2_ID)] - h2h[(PLAYER2_ID, PLAYER1_ID)]
    output["H2H_SURFACE_DIFF"] = h2h_surface[SURFACE][(PLAYER1_ID, PLAYER2_ID)] - h2h_surface[SURFACE][(PLAYER2_ID, PLAYER1_ID)]

    for k in [3, 10, 25, 50, 100]:
        ############## Last K Matches Won ##############
        if len(last_k_matches[PLAYER1_ID]) >= k and len(last_k_matches[PLAYER2_ID]) >= k:
            # Calculate wins in the last k matches
            output["WIN_LAST_"+str(k)+"_DIFF"] = sum(list(last_k_matches[PLAYER1_ID])[-k:])-sum(list(last_k_matches[PLAYER2_ID])[-k:])
        else:
            output["WIN_LAST_"+str(k)+"_DIFF"] = 0
        
        ############## ELO GRAD ##############
        # Calculate gradient BEFORE match
        if len(elo_grad_players[PLAYER1_ID]) >= k and len(elo_grad_players[PLAYER2_ID]) >= k:
            elo_grad_p1 = list(elo_grad_players[PLAYER1_ID])[-k:]
            elo_grad_p2 = list(elo_grad_players[PLAYER2_ID])[-k:]
            slope_1 = np.polyfit(np.arange(len(elo_grad_p1)), np.array(elo_grad_p1), 1)[0]
            slope_2 = np.polyfit(np.arange(len(elo_grad_p2)), np.array(elo_grad_p2), 1)[0]
            output["ELO_GRAD_LAST_"+str(k)+"_DIFF"] = slope_1-slope_2
        else:
            output["ELO_GRAD_LAST_"+str(k)+"_DIFF"] = 0

        ############# Various Other Statistics #############
        # Serve Stats
        output["P_ACE_LAST_"+str(k)+"_DIFF"] = mean(list(last_k_matches_stats[PLAYER1_ID]["p_ace"])[-k:])-mean(list(last_k_matches_stats[PLAYER2_ID]["p_ace"])[-k:])
        output["P_DF_LAST_"+str(k)+"_DIFF"] = mean(list(last_k_matches_stats[PLAYER1_ID]["p_df"])[-k:])-mean(list(last_k_matches_stats[PLAYER2_ID]["p_df"])[-k:])
        output["P_1ST_IN_LAST_"+str(k)+"_DIFF"] = mean(list(last_k_matches_stats[PLAYER1_ID]["p_1stIn"])[-k:])-mean(list(last_k_matches_stats[PLAYER2_ID]["p_1stIn"])[-k:])
        output["P_1ST_WON_LAST_"+str(k)+"_DIFF"] = mean(list(last_k_matches_stats[PLAYER1_ID]["p_1stWon"])[-k:])-mean(list(last_k_matches_stats[PLAYER2_ID]["p_1stWon"])[-k:])
        output["P_2ND_WON_LAST_"+str(k)+"_DIFF"] = mean(list(last_k_matches_stats[PLAYER1_ID]["p_2ndWon"])[-k:])-mean(list(last_k_matches_stats[PLAYER2_ID]["p_2ndWon"])[-k:])
        output["P_BP_SAVED_LAST_"+str(k)+"_DIFF"] = mean(list(last_k_matches_stats[PLAYER1_ID]["p_bpSaved"])[-k:])-mean(list(last_k_matches_stats[PLAYER2_ID]["p_bpSaved"])[-k:])
        
        # Return Stats
        output["P_RPW_LAST_"+str(k)+"_DIFF"] = (mean(list(last_k_matches_stats[PLAYER1_ID]["p_rpw"])[-k:])-mean(list(last_k_matches_stats[PLAYER2_ID]["p_rpw"])[-k:]))
        output["P_RET_1ST_WON_LAST_"+str(k)+"_DIFF"] = (mean(list(last_k_matches_stats[PLAYER1_ID]["p_ret1stWon"])[-k:])-mean(list(last_k_matches_stats[PLAYER2_ID]["p_ret1stWon"])[-k:]))
        output["P_RET_2ND_WON_LAST_"+str(k)+"_DIFF"] = (mean(list(last_k_matches_stats[PLAYER1_ID]["p_ret2ndWon"])[-k:])-mean(list(last_k_matches_stats[PLAYER2_ID]["p_ret2ndWon"])[-k:]))
        output["P_BP_CONV_LAST_"+str(k)+"_DIFF"] = (mean(list(last_k_matches_stats[PLAYER1_ID]["p_bpConv"])[-k:])-mean(list(last_k_matches_stats[PLAYER2_ID]["p_bpConv"])[-k:]))
        output["P_RET_ACE_AGAINST_LAST_"+str(k)+"_DIFF"] = (mean(list(last_k_matches_stats[PLAYER1_ID]["p_retAceAgainst"])[-k:])-mean(list(last_k_matches_stats[PLAYER2_ID]["p_retAceAgainst"])[-k:]))
        
        # Other Stats
        output["P_TOTAL_PTS_WON_LAST_"+str(k)+"_DIFF"] = (mean(list(last_k_matches_stats[PLAYER1_ID]["p_totalPtsWon"])[-k:])-mean(list(last_k_matches_stats[PLAYER2_ID]["p_totalPtsWon"])[-k:]))
        output["DOMINANCE_RATIO_LAST_"+str(k)+"_DIFF"] = (mean(list(last_k_matches_stats[PLAYER1_ID]["dominance_ratio"])[-k:])-mean(list(last_k_matches_stats[PLAYER2_ID]["dominance_ratio"])[-k:]))

    return output

if __name__ == '__main__':
    pass