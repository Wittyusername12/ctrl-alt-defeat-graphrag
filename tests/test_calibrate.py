"""calibrate.py on the hand-worked case written into the Phase 5 plan (tracker section 3, "Phase 5 cases").

Two made-up reports: gold R1 {T1036, T1055, G0129} and R2 {S0013, TA0011}; seven candidate links, two of them
wrong (T1027 in R1 and T1129 in R2). Worked by hand: at (τ 0.40, c 0) TP 5, FP 2, FN 0, F1 0.833; the best F1,
0.909, comes at (0.40, 0.4) and (0.42, 0.4), and the tie rule picks τ 0.42, c 0.4.
"""
import calibrate
import link
from schemas import GROUP, SOFTWARE, TACTIC, TECHNIQUE

LABELS = {"G0129": GROUP, "T1036": TECHNIQUE, "T1027": TECHNIQUE, "T1055": TECHNIQUE, "S0013": SOFTWARE,
          "T1129": TECHNIQUE, "TA0011": TACTIC}


def entry(attack_id):
    return link.Entry(attack_id, LABELS[attack_id], attack_id, "", None)


def exact(mention_id, attack_id, confidence):
    return {"mention_id": mention_id, "model_confidence": confidence, "exact": entry(attack_id),
            "exact_method": "exact", "best": None, "best_cosine": None, "closest": []}


def by_meaning(mention_id, attack_id, cosine, confidence):
    return {"mention_id": mention_id, "model_confidence": confidence, "exact": None, "exact_method": "no exact match",
            "best": entry(attack_id), "best_cosine": cosine, "closest": []}


ROWS = [("R1", exact("a", "G0129", 1.0)), ("R1", by_meaning("b", "T1036", 0.55, 0.9)),
        ("R1", by_meaning("c", "T1027", 0.45, 0.6)), ("R1", exact("d", "T1055", 0.4)),
        ("R2", exact("e", "S0013", 1.0)), ("R2", exact("f", "T1129", 0.3)),
        ("R2", by_meaning("g", "TA0011", 0.42, 0.95))]
GOLD = {"R1": {"T1036", "T1055", "G0129"}, "R2": {"S0013", "TA0011"}}


def test_the_candidates_are_the_seen_cosines_and_confidences():
    taus, cutoffs = calibrate.candidate_values(ROWS)
    assert taus == [0.40, 0.42, 0.45, 0.55, 1.01]
    assert cutoffs == [0.0, 0.3, 0.4, 0.6, 0.9, 0.95, 1.0]


def test_the_hand_worked_calibration_case():
    taus, cutoffs = calibrate.candidate_values(ROWS)
    table = calibrate.grid(ROWS, GOLD, LABELS.__getitem__, {}, taus, cutoffs)
    assert len(table) == 35
    start = next(row for row in table if (row["tau"], row["cutoff"]) == (0.40, 0.0))
    assert (start["tp"], start["fp"], start["fn"], round(start["f1"], 3)) == (5, 2, 0, 0.833)
    best = calibrate.choose(table)
    assert (best["tau"], best["cutoff"], best["tp"], best["fp"], best["fn"], round(best["f1"], 3)) == (0.42, 0.4, 5, 1, 0, 0.909)
    ties = sorted((row["tau"], row["cutoff"]) for row in table if calibrate.f1_exact(row) == calibrate.f1_exact(best))
    assert ties == [(0.40, 0.4), (0.42, 0.4)]


def test_the_stored_threshold_lies_strictly_between_the_chosen_cosine_and_the_next_lower_one():
    taus = [0.40, 0.42, 0.45, 0.55, 1.01]
    assert calibrate.stored_threshold(0.42, taus) == 0.41
    assert calibrate.stored_threshold(0.55, taus) == 0.5
    assert calibrate.stored_threshold(0.40, taus) == 0.40 and calibrate.stored_threshold(1.01, taus) == 1.01
    close = [0.40, 0.612345, 0.612349, 1.01]
    value = calibrate.stored_threshold(0.612349, close)
    assert 0.612345 < value < 0.612349 and value == 0.612347
    facts_at_choice = calibrate.facts_at(ROWS, 0.42, 0.4)
    assert calibrate.facts_at(ROWS, calibrate.stored_threshold(0.42, taus), 0.4) == facts_at_choice
