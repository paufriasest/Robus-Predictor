import pandas as pd


def select_stable_cubes(
    domains,
    mean_min,
    mean_max,
    std_min,
    std_max,
    verbose=False
):
    if not domains:
        raise ValueError("domains no puede estar vacío.")

    for i, domain in enumerate(domains):
        if "x" not in domain:
            raise ValueError(f"El dominio {i + 1} no contiene la clave 'x'.")

        if "y" not in domain:
            raise ValueError(f"El dominio {i + 1} no contiene la clave 'y'.")

        if "groups" not in domain:
            raise ValueError(f"El dominio {i + 1} no contiene la clave 'groups'.")

    base_groups = domains[0]["groups"]
    group_ids = list(base_groups.keys())

    stable_cubes = []
    red_zones = []

    for group_id in group_ids:
        domain_stats = []
        is_stable = True
        rejection_reasons = []

        for domain_index, domain in enumerate(domains):
            groups = domain["groups"]
            y = domain["y"]

            if group_id not in groups:
                is_stable = False
                rejection_reasons.append(
                    f"Dominio {domain_index + 1}: no existe el grupo {group_id}"
                )
                continue

            x_group = groups[group_id]

            if x_group.empty:
                is_stable = False
                rejection_reasons.append(
                    f"Dominio {domain_index + 1}: grupo vacío"
                )

                domain_stats.append({
                    "domain": domain_index + 1,
                    "group_id": group_id,
                    "n": 0,
                    "mean": None,
                    "std": None,
                    "mean_ok": False,
                    "std_ok": False,
                    "is_valid": False,
                    "reason": "grupo vacío"
                })

                continue

            y_group = y.loc[x_group.index]

            if isinstance(y_group, pd.DataFrame):
                if y_group.shape[1] != 1:
                    raise ValueError(
                        "y debe ser una Series o un DataFrame de una sola columna."
                    )

                y_group = y_group.iloc[:, 0]

            mean_value = y_group.mean()
            std_value = y_group.std()

            # Si el grupo tiene 1 solo elemento, pandas devuelve std = NaN.
            # Para esta lógica lo tratamos como 0.
            if pd.isna(std_value):
                std_value = 0.0

            mean_ok = mean_min <= mean_value <= mean_max
            std_ok = std_min <= std_value <= std_max

            group_valid = mean_ok and std_ok

            reason = []

            if not mean_ok:
                reason.append(
                    f"mean fuera de rango: {mean_value} no está entre {mean_min} y {mean_max}"
                )

            if not std_ok:
                reason.append(
                    f"std fuera de rango: {std_value} no está entre {std_min} y {std_max}"
                )

            if not group_valid:
                is_stable = False
                rejection_reasons.append(
                    f"Dominio {domain_index + 1}: " + "; ".join(reason)
                )

            domain_stats.append({
                "domain": domain_index + 1,
                "group_id": group_id,
                "n": len(y_group),
                "mean": mean_value,
                "std": std_value,
                "mean_ok": mean_ok,
                "std_ok": std_ok,
                "is_valid": group_valid,
                "reason": "; ".join(reason) if reason else "OK"
            })

        valid_means = [
            stat["mean"]
            for stat in domain_stats
            if stat["mean"] is not None
        ]

        prediction_value = (
            sum(valid_means) / len(valid_means)
            if valid_means
            else None
        )

        cube_info = {
            "group_id": group_id,
            "prediction_value": prediction_value,
            "stats": domain_stats,
            "is_stable": is_stable,
            "rejection_reasons": rejection_reasons
        }

        if is_stable:
            stable_cubes.append(cube_info)

            if verbose:
                print(f"[STABLE] Grupo {group_id} | pred={prediction_value}")

        else:
            red_zones.append(cube_info)

            if verbose:
                print(f"[RED ZONE] Grupo {group_id}")
                for reason in rejection_reasons:
                    print(f"  - {reason}")

    return stable_cubes, red_zones