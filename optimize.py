import os
import random
import logging

import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import Series, DataFrame
from pydantic import BaseModel
import cvxpy as cp

logger = logging.getLogger(__name__)

POSITIONS = ["GKP", "DEF", "MID", "FWD"]
BASE_FORMATION = (2, 5, 5, 3)
FORMATIONS = [
    (1, 3, 4, 3),
    (1, 3, 5, 2),
    (1, 4, 3, 3),
    (1, 4, 4, 2),
    (1, 4, 5, 1),
    (1, 5, 2, 3),
    (1, 5, 3, 2),
    (1, 5, 4, 1),
]
CLUBS = [
    "ARS",
    "AVL",
    "BHA",
    "BOU",
    "BRE",
    "CHE",
    "CRY",
    "EVE",
    "FUL",
    "LEE",
    "LEI",
    "LIV",
    "MCI",
    "MUN",
    "NEW",
    "NFO",
    "SOU",
    "TOT",
    "WHU",
    "WOL",
]


class PlayerDataframeSchema(pa.SchemaModel):
    name: Series[str]
    is_available: Series[bool]
    in_squad: Series[bool]
    expected_points: Series[float]
    value: Series[float]
    position: Series[str]
    team: Series[str]


class OptimizationResult(BaseModel):
    expected_team_points: int = 0
    formation: tuple[int, int, int, int] | None = None
    itb_change: int = 0
    transfers: int = 0
    transfer_cost: int = 0
    new_squad: DataFrame[PlayerDataframeSchema]
    value: int


def optimize(
    players: DataFrame[PlayerDataframeSchema],
    itb: int = 0,
    ft: int = 1,
    margin: float = 0.0,
    filler_count: int = 4,
):
    """
    Find squad with maximum expected points based on individual predicted
    points, the current squad, formation and other rules and the cost of
    transfers.

    Params
    ------

    players: pandas DataFrame
        Overall player dataset with predicted points.

    itb: int
        Funds "in the bank" (default: 0).

    ft: int
        Number of free transfers available (default: 1)

    margin: int
        Modifier for transfer cost to reduce squad turnover. Transfers
        after free transfers cost (4 + margin) points (default: 0).

    filler_count: int (default: 0)
        Number of non-playing bench players (presumably cheap w/ low
        point potential).

    Returns: tuple
        new squad, transfer count, transfers cost, change in ITB funds
    """

    old_squad = players.query("in_squad")
    old_squad_value = np.dot(players["value"], players["in_squad"])

    clubs = pd.get_dummies(players["team"], drop_first=False)[CLUBS]
    positions = pd.get_dummies(players["position"], drop_first=False)[POSITIONS]
    data = pd.concat([players, positions, clubs], axis=1).query("is_available")

    optimum = OptimizationResult(new_squad=old_squad, value=old_squad_value)

    for formation in FORMATIONS:
        starters = cp.Variable(len(data.index), boolean=True)
        bench = cp.Variable(len(data.index), boolean=True)
        filler = cp.Variable(len(data.index), boolean=True)

        expected_points = data["expected_points"].values @ (starters + bench)
        transfer_count = 15 - data["in_squad"].values @ (starters + bench + filler)
        transfer_cost = cp.maximum((transfer_count - ft) * (4 + margin), 0)

        objective = cp.Maximize(expected_points - transfer_cost)

        constraints = [
            bench >= 0,
            bench <= 1,
            starters >= 0,
            starters <= 1,
            filler >= 0,
            filler <= 1,
            # Must not cost more than old squad value + ITB
            data["value"].values @ (starters + bench + filler) - old_squad_value <= itb,
            # New starters must adhere to starting formation
            data[POSITIONS].values.T @ starters == np.array(formation),
            # Total squad must adhere to basic position limits
            data[POSITIONS].values.T @ (starters + bench + filler)
            == np.array(BASE_FORMATION),
            # Max. 3 players from every club
            data[CLUBS].values.T @ (starters + bench + filler)
            <= np.full(len(CLUBS), 3),
            #  One player can only be either starter or on bench/fill
            starters + bench + filler <= np.full(len(data.index), 1),
            # Filler count
            filler @ np.full(len(data.index), 1) == filler_count,
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve()
        if prob.value is None:
            logger.warning("Cannot optimize for {}".format(formation))
        else:
            expected_team_points = np.round(prob.value, 0)

            data["starter"] = np.round(starters.value, 0).astype("bool")
            data["sub"] = np.round(bench.value, 0).astype("bool")
            data["filler"] = np.round(filler.value, 0).astype("bool")
            new_squad = data[data["starter"] | data["sub"] | data["filler"]][
                [
                    "name",
                    "team",
                    "position",
                    "value",
                    "starter",
                    "expected_points",
                    "is_available",
                ]
            ]
            itb_change = np.round(old_squad_value - new_squad["value"].sum(), 1)

            new_squad["rank"] = new_squad.groupby("starter")[
                "expected_points"
            ].transform(lambda df: df.rank(ascending=False))

            new_squad["captain"] = (new_squad["rank"] == 1) & (new_squad["starter"])
            new_squad["vice_captain"] = (new_squad["rank"] == 2) & (
                new_squad["starter"]
            )

            better_result = expected_team_points > optimum.expected_team_points
            same_but_cheaper = (
                expected_team_points == optimum.expected_team_points
                and new_squad["value"].sum() < optimum.value
            )

            if better_result or same_but_cheaper:
                optimum = OptimizationResult(
                    formation=formation,
                    expected_team_points=expected_team_points,
                    itb_change=itb_change,
                    transfers=transfer_count.value,
                    transfer_cost=transfer_cost.value,
                    new_squad=new_squad.drop("expected_points", axis=1),
                    value=new_squad["value"].sum(),
                )

                logger.info(f"Found better solution: {optimum}")

    return optimum
