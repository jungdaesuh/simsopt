from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FrontierLaneSpec:
    lane_id: str
    scalarization_type: str
    scalarization_params: dict[str, float]
    iotas_weight: float
    frontier_volume_weight: float
    res_weight: float
    lane_budget: int | None

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_json_dict(
        cls,
        payload: dict[str, object],
    ) -> FrontierLaneSpec:
        scalarization_params_payload = payload.get("scalarization_params", {})
        return cls(
            lane_id=str(payload["lane_id"]),
            scalarization_type=str(payload["scalarization_type"]),
            scalarization_params={
                str(key): float(value)
                for key, value in scalarization_params_payload.items()
            },
            iotas_weight=float(payload["iotas_weight"]),
            frontier_volume_weight=float(payload["frontier_volume_weight"]),
            res_weight=float(payload["res_weight"]),
            lane_budget=None
            if payload.get("lane_budget") is None
            else int(payload["lane_budget"]),
        )


def generate_multilane_local_specs(
    *,
    num_lanes: int,
    iotas_weight: float,
    frontier_volume_weight: float | None,
    res_weight: float,
    lane_budget: int | None,
) -> list[FrontierLaneSpec]:
    if num_lanes <= 0:
        raise ValueError("--frontier-num-lanes must be positive")

    base_volume_weight = float(iotas_weight if frontier_volume_weight is None else frontier_volume_weight)
    total_reward_weight = float(iotas_weight) + base_volume_weight
    if num_lanes == 1:
        shares = [0.5]
    else:
        min_share = 0.2
        max_share = 0.8
        step = (max_share - min_share) / float(num_lanes - 1)
        shares = [min_share + step * index for index in range(num_lanes)]

    lane_specs: list[FrontierLaneSpec] = []
    for index, iota_share in enumerate(shares):
        volume_share = 1.0 - iota_share
        lane_specs.append(
            FrontierLaneSpec(
                lane_id=f"lane_{index + 1:02d}",
                scalarization_type="weight_schedule_v1",
                scalarization_params={
                    "iota_share": float(iota_share),
                    "volume_share": float(volume_share),
                },
                iotas_weight=total_reward_weight * float(iota_share),
                frontier_volume_weight=total_reward_weight * float(volume_share),
                res_weight=float(res_weight),
                lane_budget=lane_budget,
            )
        )
    return lane_specs
