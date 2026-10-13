# assigment-2
# Assignment 2 — Adversarial Search and Game Trees

TREE = {
    "A": ("MAX", ["B", "C"]),
    "B": ("MIN", ["D", "E"]),
    "C": ("MIN", ["F", "G"]),
    "D": ("MAX", ["H", "I"]),
    "E": ("MAX", ["J", "K"]),
    "F": ("MAX", ["L", "M"]),
    "G": ("MAX", ["N", "O"]),
    "H": 3,
    "I": 12,
    "J": 8,
    "K": 2,
    "L": 4,
    "M": 6,
    "N": 14,
    "O": 5,
}


def subtree_nodes(node):
    """Түйінді және оның барлық ұрпағын қайтарады."""
    item = TREE[node]

    if isinstance(item, int):
        return [node]

    nodes = [node]

    for child in item[1]:
        nodes.extend(subtree_nodes(child))

    return nodes


def search(pruning=False, reverse=False):
    visited = []
    pruned = []
    trace = []
    values = {}
    moves = {}

    def visit(node, alpha, beta):
        visited.append(node)
        item = TREE[node]

        # Жапырақ болса, дайын ұпайды қайтарамыз.
        if isinstance(item, int):
            values[node] = item
            return item

        role, children = item

        if reverse:
            children = children[::-1]

        alpha_before = alpha
        beta_before = beta

        if role == "MAX":
            value = -float("inf")
        else:
            value = float("inf")

        best_move = None

        for index, child in enumerate(children):
            score = visit(child, alpha, beta)

            if role == "MAX":
                if score > value:
                    value = score
                    best_move = child
            else:
                if score < value:
                    value = score
                    best_move = child

            if pruning:
                if role == "MAX":
                    alpha = max(alpha, value)
                else:
                    beta = min(beta, value)

                # Қалған тармақтарды тексерудің қажеті жоқ.
                if alpha >= beta:
                    for skipped_child in children[index + 1:]:
                        pruned.extend(subtree_nodes(skipped_child))
                    break

        values[node] = value
        moves[node] = best_move

        trace.append({
            "node": node,
            "role": role,
            "alpha_before": alpha_before,
            "beta_before": beta_before,
            "value": value,
            "alpha_after": alpha,
            "beta_after": beta,
        })

        return value

    root_value = visit("A", -float("inf"), float("inf"))

    return {
        "value": root_value,
        "move": moves["A"],
        "count": len(visited),
        "visited": visited,
        "pruned": pruned,
        "trace": trace,
        "values": values,
        "moves": moves,
    }


def minimax(reverse=False):
    return search(pruning=False, reverse=reverse)


def alpha_beta(reverse=False):
    return search(pruning=True, reverse=reverse)


def evaluation(features, weights):
    """Белгілердің салмақталған қосындысы."""
    return sum(
        feature * weight
        for feature, weight in zip(features, weights)
    )


def main():
    # A. MINIMAX
    print("\nA. MINIMAX")

    mm = minimax()

    for node in ["D", "E", "F", "G", "B", "C", "A"]:
        role, children = TREE[node]
        child_values = [mm["values"][child] for child in children]

        print(
            f"{node}: {role} {child_values} "
            f"= {mm['values'][node]}"
        )

    path = ["A"]

    while path[-1] in mm["moves"]:
        path.append(mm["moves"][path[-1]])

    print("Таңдалған жүріс:", mm["move"])
    print("Түбір мәні:", mm["value"])
    print("Оңтайлы жол:", " → ".join(path))

    # B. ALPHA–BETA
    print("\nB. ALPHA–BETA: СОЛДАН ОҢҒА")

    ab = alpha_beta()

    print("Түйін | Түрі | α бастапқы | β бастапқы | Мән | α соңғы | β соңғы")

    for row in ab["trace"]:
        print(
            f"{row['node']} | {row['role']} | "
            f"{row['alpha_before']} | {row['beta_before']} | "
            f"{row['value']} | "
            f"{row['alpha_after']} | {row['beta_after']}"
        )

    print("Қаралған түйіндер:", ", ".join(ab["visited"]))
    print("Қиылған түйіндер:", ", ".join(ab["pruned"]))
    print("Қаралған түйін саны:", ab["count"])

    # C. PYTHON ЭКСПЕРИМЕНТІ
    print("\nC. ЕКІ БАҒЫТТЫ САЛЫСТЫРУ")

    experiments = [
        ("Солдан оңға", False),
        ("Оңнан солға", True),
        ("Солдан оңға қайта тексеру", False),
    ]

    for label, reverse in experiments:
        mm_result = minimax(reverse)
        ab_result = alpha_beta(reverse)

        assert mm_result["value"] == ab_result["value"] == 8
        assert mm_result["move"] == ab_result["move"] == "B"
        assert mm_result["count"] == 15

        expected_count = 14 if reverse else 12
        assert ab_result["count"] == expected_count
        assert len(ab_result["pruned"]) == 15 - expected_count

        print(f"\n{label}:")
        print("Тереңдік: 3 қабырға, 4 деңгей")
        print("Minimax түйін саны:", mm_result["count"])
        print("Alpha–Beta түйін саны:", ab_result["count"])
        print("Қиылған түйін саны:", len(ab_result["pruned"]))
        print("Қиылған түйіндер:", ", ".join(ab_result["pruned"]))
        print("Өту реті:", ", ".join(ab_result["visited"]))
        print("Шешім:", ab_result["move"], "Мән:", ab_result["value"])

    # D. БАҒАЛАУ ФУНКЦИЯСЫ
    print("\nD. БАҒАЛАУ ФУНКЦИЯСЫ")

    # Белгілер: материал, позиция, қозғалыс мүмкіндігі.
    # X және Y — негізгі ағаштан бөлек шартты күйлер.
    states = {
        "X": (4, 1, 0),
        "Y": (0, 0, 6),
    }

    initial_weights = (0.5, 0.3, 0.2)

    new_mobility_weight = 0.5
    factor = (1 - new_mobility_weight) / (0.5 + 0.3)

    changed_weights = (
        0.5 * factor,
        0.3 * factor,
        new_mobility_weight,
    )

    assert changed_weights == (0.3125, 0.1875, 0.5)
    assert abs(sum(changed_weights) - 1) < 1e-12

    for label, weights in [
        ("Бастапқы", initial_weights),
        ("Өзгертілген", changed_weights),
    ]:
        scores = {
            name: evaluation(features, weights)
            for name, features in states.items()
        }

        best_state = max(scores, key=scores.get)

        print(f"\n{label} салмақтар: {weights}")
        print("X ұпайы:", scores["X"])
        print("Y ұпайы:", scores["Y"])
        print("Таңдалған күй:", best_state)

    # E. ТЕХНИКАЛЫҚ ТАЛДАУ
    print("\nE. ТҮЙІН САНЫ БОЙЫНША ҮНЕМ")

    for reverse in (False, True):
        mm_result = minimax(reverse)
        ab_result = alpha_beta(reverse)

        saved = mm_result["count"] - ab_result["count"]
        percentage = saved / mm_result["count"] * 100

        direction = "Оңнан солға" if reverse else "Солдан оңға"

        print(
            f"{direction}: {saved} түйін үнемделді "
            f"({percentage:.2f}%)"
        )

    print("Орындалу уақыты өлшенген жоқ.")
    print("Екі бағытта да түбір мәні 8, таңдалған жүріс B.")

    # F. ФИЛОСОФИЯЛЫҚ ТАЛДАУҒА ТЕХНИКАЛЫҚ ДӘЛЕЛ
    print("\nF. ТЕХНИКАЛЫҚ ДӘЛЕЛДІ ТЕКСЕРУ")

    results = [
        minimax(),
        alpha_beta(),
        alpha_beta(reverse=True),
    ]

    assert all(
        result["value"] == 8 and result["move"] == "B"
        for result in results
    )

    print("Барлық әдісте: A = 8, таңдау = B.")
    print("Қаралған түйіндер:", [result["count"] for result in results])
    print(
        "Бұл нәтиже берілген ағаштағы оңтайлы таңдауды көрсетеді. "
        "Ол жалпы интеллектті, түсінуді, шығармашылықты "
        "немесе моральдық жауапкершілікті өздігінен дәлелдемейді."
    )


if __name__ == "__main__":
    main()
