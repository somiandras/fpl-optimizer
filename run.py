from fpl import get_player_data, get_fixtures
from optimize import optimize

df = get_player_data(my_team_path="sample_data/team_data.json")

results = optimize(df, 20, 15)

print(results)
