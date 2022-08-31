from preprocess import get_player_data, get_fixtures
from optimize import optimize

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

df = get_player_data(my_team_path="sample_data/team_data.json")

results = optimize(df, 20, 15)

print(results)
