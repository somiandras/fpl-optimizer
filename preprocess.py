import json

import requests
import pandas as pd
import numpy as np

from schema import Fixture, PlayerFixtures, Summary, MyTeam

STATIC_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
PLAYER_URL = "https://fantasy.premierleague.com/api/element-summary/{player_id}/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
ME = "https://fantasy.premierleague.com/api/my-team/4001050/"


def get_summary():
    r = requests.get(STATIC_URL)
    return Summary.parse_obj(r.json())


def get_player_fixtures(player_id: int):
    r = requests.get(PLAYER_URL.format(player_id=player_id))
    return PlayerFixtures.parse_obj(r.json())


def get_my_team(path: str):
    return MyTeam.parse_file(path).picks


def get_fixtures():
    r = requests.get(FIXTURES_URL)
    return [Fixture.parse_obj(fixture) for fixture in r.json()]


def get_difficulty_multipliers(at_gw: int):
    fixtures = pd.DataFrame.from_records([fixture.dict() for fixture in get_fixtures()])

    return (
        pd.concat(
            [
                fixtures[["id", "event", "team_h", "team_h_difficulty"]].rename(
                    columns={"team_h": "team", "team_h_difficulty": "difficulty"}
                ),
                fixtures[["id", "event", "team_a", "team_a_difficulty"]].rename(
                    columns={"team_a": "team", "team_a_difficulty": "difficulty"}
                ),
            ]
        )
        .query("@at_gw <= event < @at_gw + 2")
        .groupby("team")["difficulty"]
        .sum()
        .pipe(lambda x: x.div(x.mean()))
        .rename("difficulty_multiplier")
    )


def get_player_data(my_team_path: str, at_gw: int):
    summary = get_summary()

    my_team = pd.DataFrame.from_records(
        [
            pick.dict(include={"element": True, "selling_price": True})
            for pick in get_my_team(path=my_team_path)
        ],
        index="element",
    )

    player_types = pd.DataFrame.from_records(
        [
            element_type.dict(include={"id": True, "singular_name_short": True})
            for element_type in summary.element_types
        ],
        index="id",
    ).rename(columns={"singular_name_short": "position"})

    teams = (
        pd.DataFrame.from_records(
            [
                team.dict(include={"id": True, "short_name": True})
                for team in summary.teams
            ],
            index="id",
        )
        .rename(columns={"short_name": "team"})
        .join(get_difficulty_multipliers(at_gw))
    )

    players = (
        pd.DataFrame.from_records(
            [
                element.dict(
                    include={
                        "element_type": True,
                        "id": True,
                        "now_cost": True,
                        "total_points": True,
                        "team": True,
                        "status": True,
                        "web_name": True,
                    }
                )
                for element in summary.elements
            ],
            index="id",
        )
        .join(player_types, on="element_type")
        .join(teams, on="team", lsuffix="_element")
        .join(my_team)
        .assign(
            value=lambda x: x["selling_price"].fillna(x["now_cost"]),
            expected_points=lambda x: x["total_points"]
            / np.sqrt(x["difficulty_multiplier"]),
            in_squad=lambda x: x["selling_price"].notnull(),
            is_available=lambda x: x["status"] == "a",
        )
        .drop(
            [
                "element_type",
                "team_element",
                "now_cost",
                "selling_price",
                "total_points",
                "difficulty_multiplier",
                "status",
            ],
            axis=1,
        )
        .rename(columns={"web_name": "name"})
    )

    return players
