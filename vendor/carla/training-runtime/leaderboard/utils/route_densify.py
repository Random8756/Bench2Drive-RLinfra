from typing import Iterable, List, Sequence, Tuple


def densify_xyz_points(
    points: Sequence[Sequence[float]],
    insert_num: int,
) -> List[Tuple[float, float, float]]:
    if insert_num < 0:
        raise ValueError("insert_num must be non-negative")

    dense_points = [tuple(map(float, point)) for point in points]
    if insert_num == 0 or len(dense_points) <= 1:
        return dense_points

    out: List[Tuple[float, float, float]] = []
    denom = insert_num + 1

    for idx in range(len(dense_points) - 1):
        ax, ay, az = dense_points[idx]
        bx, by, bz = dense_points[idx + 1]
        out.append((ax, ay, az))

        for step in range(1, insert_num + 1):
            t = step / denom
            out.append((
                ax + (bx - ax) * t,
                ay + (by - ay) * t,
                az + (bz - az) * t,
            ))

    out.append(dense_points[-1])
    return out


def densify_locations(
    locations: Iterable[object],
    insert_num: int,
    location_cls=None,
):
    locations = list(locations)
    if not locations:
        return []

    if location_cls is None:
        location_cls = type(locations[0])

    dense_xyz = densify_xyz_points(
        [(float(loc.x), float(loc.y), float(loc.z)) for loc in locations],
        insert_num,
    )
    return [_build_location(location_cls, x, y, z) for x, y, z in dense_xyz]


def _build_location(location_cls, x: float, y: float, z: float):
    try:
        return location_cls(x=x, y=y, z=z)
    except TypeError:
        return location_cls(x, y, z)
