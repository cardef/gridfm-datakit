import numpy as np
import pandas as pd

from gridfm_datakit.forge.graph import compile_scenario


def test_compile_scenario_preserves_ids_and_builds_labels():
    bus = pd.DataFrame(
        {
            "scenario": [7, 7],
            "bus": [10, 20],
            "Pd": [1.0, 2.0],
            "Qd": [0.1, 0.2],
            "Pg": [2.0, 1.0],
            "Qg": [0.0, 0.0],
            "Vm": [1.0, 1.0],
            "Va": [0.0, 0.1],
            "vn_kv": [220.0, 220.0],
            "min_vm_pu": [0.9, 0.9],
            "max_vm_pu": [1.1, 1.1],
            "GS": [0.0, 0.0],
            "BS": [0.0, 0.0],
        }
    )
    branch = pd.DataFrame(
        {
            "scenario": [7],
            "idx": [3],
            "from_bus": [10],
            "to_bus": [20],
            "pf": [99.95],
            "qf": [0.0],
            "pt": [-99.95],
            "qt": [0.0],
            "r": [0.01],
            "x": [0.1],
            "b": [0.0],
            "tap": [1.0],
            "shift": [0.0],
            "ang_min": [-30.0],
            "ang_max": [30.0],
            "rate_a": [100.0],
            "br_status": [1],
        }
    )
    gen = pd.DataFrame(
        {
            "scenario": [7],
            "idx": [0],
            "bus": [10],
            "p_mw": [100.0],
            "q_mvar": [0.0],
            "min_p_mw": [0.0],
            "max_p_mw": [100.0],
            "min_q_mvar": [-50.0],
            "max_q_mvar": [50.0],
            "cp0_eur": [0.0],
            "cp1_eur_per_mw": [10.0],
            "cp2_eur_per_mw2": [0.0],
            "in_service": [1],
            "is_slack_gen": [1],
        }
    )

    sample = compile_scenario(7, bus, branch, gen, active_tolerance=1e-3)

    np.testing.assert_array_equal(sample.node_ids, [10, 20])
    np.testing.assert_array_equal(sample.edge_index, [[0], [1]])
    np.testing.assert_array_equal(sample.generator_bus, [0])
    np.testing.assert_array_equal(sample.labels["branch_active"], [1])
    np.testing.assert_array_equal(sample.labels["gen_p_upper_active"], [1])
