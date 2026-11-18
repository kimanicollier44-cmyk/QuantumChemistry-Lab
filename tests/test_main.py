"""Tests for QuantumChemistry-Lab: Jacobi solver, CI, SCF, curves, gates."""
import math

import pytest

import main as m


# ---- Jacobi eigensolver
def test_jacobi_known_2x2():
    vals, _ = m._sorted_eig([[2.0, 1.0], [1.0, 2.0]])
    assert vals[0] == pytest.approx(1.0, abs=1e-10)
    assert vals[1] == pytest.approx(3.0, abs=1e-10)


def test_jacobi_diagonal_matrix():
    vals, _ = m._sorted_eig([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0],
                             [0.0, 0.0, 5.0]])
    assert vals == [2.0, 3.0, 5.0]


def test_jacobi_eigenvectors_orthonormal():
    _, vecs = m._sorted_eig([[4.0, 1.0, 2.0], [1.0, 3.0, 0.5],
                             [2.0, 0.5, 5.0]])
    for i in range(3):
        for j in range(3):
            dot = sum(vecs[r][i] * vecs[r][j] for r in range(3))
            assert dot == pytest.approx(1.0 if i == j else 0.0, abs=1e-9)


def test_jacobi_reconstruction():
    mat = [[4.0, 1.0, 2.0], [1.0, 3.0, 0.5], [2.0, 0.5, 5.0]]
    vals, vecs = m._sorted_eig(mat)
    for i in range(3):
        for j in range(3):
            rebuilt = sum(vecs[i][k] * vals[k] * vecs[j][k] for k in range(3))
            assert rebuilt == pytest.approx(mat[i][j], abs=1e-9)


# ---- labelled integral model
def test_overlap_sto3g_class():
    assert m.overlap_h2(1.4) == pytest.approx(0.6593, abs=0.001)


def test_overlap_limits():
    assert m.overlap_h2(0.2) > 0.95
    assert m.overlap_h2(8.0) < 0.01


def test_vnn_exact():
    assert m.h2_integrals(2.0)["Vnn"] == 0.5
    assert m.heh_integrals(2.0)["Vnn"] == 1.0


def test_integrals_dissociation_limits():
    far = m.h2_integrals(12.0)
    for i in range(2):
        for j in range(2):
            assert far["J"][i][j] == pytest.approx(m.U_HALF, abs=0.02)
            assert far["K"][i][j] == pytest.approx(m.U_HALF, abs=0.02)


# ---- determinant CI
def test_determinant_counts():
    assert len(m.ms0_determinants(2)) == 4
    assert len(m.ms0_determinants(3)) == 9


def test_ci_matrix_symmetric():
    H = m.ci_matrix(m.h2_integrals(1.4), 2)
    for i in range(4):
        for j in range(4):
            assert H[i][j] == pytest.approx(H[j][i], abs=1e-12)


def test_parity_blocks_open_shell():
    integ = m.h2_integrals(1.4)
    H = m.ci_matrix(integ, 2)
    dets = m.ms0_determinants(2)
    g2 = dets.index(((0, 0), (0, 1)))
    open_det = dets.index(((0, 0), (1, 1)))
    assert H[g2][open_det] == pytest.approx(0.0, abs=1e-12)


def test_pair_hopping_is_exchange():
    integ = m.h2_integrals(1.4)
    H = m.ci_matrix(integ, 2)
    dets = m.ms0_determinants(2)
    g2 = dets.index(((0, 0), (0, 1)))
    u2 = dets.index(((1, 0), (1, 1)))
    assert H[g2][u2] == pytest.approx(integ["K"][0][1], abs=1e-12)


def test_hf_anchor_energy():
    assert -1.125 < m.hf_energy_mo(m.h2_integrals(1.4)) < -1.108


def test_exact_anchor_energy():
    assert -1.150 < m.fci(m.h2_integrals(1.4), 2)["energy"] < -1.130


def test_correlation_anchor_range():
    corr = m.correlation_energy(m.h2_integrals(1.4), 2)
    assert -0.035 < corr < -0.015


def test_variational_principle_on_grid():
    for R in m.CURVE_GRID:
        integ = m.h2_integrals(R)
        assert m.hf_energy_mo(integ) >= m.fci(integ, 2)["energy"] - 1e-12


def test_ground_state_mostly_g2():
    exact = m.fci(m.h2_integrals(1.4), 2)
    dets = exact["determinants"]
    g2 = dets.index(((0, 0), (0, 1)))
    assert 0.95 < exact["ground_vector"][g2] ** 2 < 1.0


# ---- SCF
def test_scf_converges_and_matches_mo_formula():
    scf = m.scf_h2(1.4)
    assert scf["converged"] and scf["iterations"] <= 25
    assert scf["energy"] == pytest.approx(
        m.hf_energy_mo(m.h2_integrals(1.4)), abs=1e-9)


def test_scf_orbital_is_bonding_mo():
    scf = m.scf_h2(1.4)
    expect = 1.0 / math.sqrt(2 * (1 + m.overlap_h2(1.4)))
    assert scf["coeff"][0] == pytest.approx(expect, abs=1e-6)
    assert scf["coeff"][0] == pytest.approx(scf["coeff"][1], abs=1e-6)


def test_scf_orbital_energy():
    integ = m.h2_integrals(1.4)
    scf = m.scf_h2(1.4)
    assert scf["eps_occ"] == pytest.approx(
        integ["h"][0] + integ["J"][0][0], abs=1e-9)


def test_ao_mo_roundtrip():
    integ = m.h2_integrals(1.4)
    back = m.mo_from_ao(m.ao_from_mo(integ, m.overlap_h2(1.4)))
    for i in range(2):
        assert back["h"][i] == pytest.approx(integ["h"][i], abs=1e-9)
        for j in range(2):
            assert back["J"][i][j] == pytest.approx(integ["J"][i][j], abs=1e-9)
            assert back["K"][i][j] == pytest.approx(integ["K"][i][j], abs=1e-9)


def test_h2plus_hf_is_exact():
    integ = m.h2_integrals(1.4)
    assert m.h2plus_exact(1.4) == pytest.approx(
        integ["h"][0] + integ["Vnn"], abs=1e-10)


# ---- bond curve
def test_curve_minimum():
    curve = m.h2_curve()
    assert 1.34 < curve["R_min_bohr"] < 1.46
    assert 0.71 < curve["R_min_angstrom"] < 0.77
    assert curve["curvature_at_min"] > 0


def test_curve_dissociation():
    curve = m.h2_curve()
    last = curve["rows"][-1]
    assert -1.01 < last["E_exact"] < -0.96
    assert last["E_HF"] - last["E_exact"] >= 0.2
    assert -0.72 < last["E_HF"] < -0.65
    assert 0.10 < curve["De"] < 0.18


# ---- HeH+
def test_heh_minimum_and_energy():
    heh = m.heh_summary()
    assert 1.30 < heh["R_min_bohr"] < 1.65
    assert -3.02 < heh["E_exact"] < -2.90
    assert heh["binding_vs_R4"] >= 0.10
    assert heh["E_HF"] >= heh["E_exact"] - 1e-12


# ---- larger toy
def test_large_toy_more_correlation():
    integ = m.h2_integrals(1.4)
    e_hf = m.hf_energy_mo(integ)
    e_small = m.fci(integ, 2)["energy"]
    e_large = m.fci(m.h2_large_integrals(), 3)["energy"]
    assert e_large < e_small - 0.005
    assert (e_large - e_hf) < (e_small - e_hf) - 0.005


def test_large_toy_same_hf():
    assert m.hf_energy_mo(m.h2_large_integrals()) == pytest.approx(
        m.hf_energy_mo(m.h2_integrals(1.4)), abs=1e-12)


# ---- basis honesty table
def test_basis_table_monotone_and_limits():
    rows = m.BASIS_TABLE
    fci_rows = [r["E_FCI"] for r in rows if r["E_FCI"] is not None]
    hf_rows = [r["E_HF"] for r in rows if r["E_HF"] is not None]
    assert all(fci_rows[i] > fci_rows[i + 1] for i in range(len(fci_rows) - 1))
    assert all(hf_rows[i] > hf_rows[i + 1] for i in range(len(hf_rows) - 1))
    assert min(hf_rows) >= -1.1336 - 1e-9
    assert min(fci_rows) >= -1.1745 - 1e-9


def test_basis_toy_row_matches_computation():
    rep = m.basis_report()
    integ = m.h2_integrals(1.4)
    assert rep["toy_row"]["E_HF"] == pytest.approx(
        m.hf_energy_mo(integ), abs=1e-9)
    assert rep["toy_row"]["E_FCI"] == pytest.approx(
        m.fci(integ, 2)["energy"], abs=1e-9)
    assert all(r["label"].startswith("LABELLED") for r in rep["rows"])


# ---- classifier
def test_classifier_scenarios():
    rep = m.classifier_report()
    assert rep["R_1.40_equilibrium"] == "bound"
    assert rep["R_2.60_stretched"] == "weak"
    assert rep["R_3.50_long"] == "weak"
    assert rep["R_6.00_apart"] == "dissociated"


def test_classifier_matches_model():
    for R in (0.8, 1.05, 1.4, 2.6, 3.5, 6.0):
        assert m.classify_bond(R) == m.model_label(*m._bond_features(R))


# ---- gates + seats
def test_hold_gates_count_and_style():
    assert len(m.HOLD_GATES) == 14
    assert all(v.startswith("HOLD") for v in m.HOLD_GATES.values())


@pytest.mark.parametrize("action", [
    "drug_discovery_claim", "weapons_chemistry_guidance",
    "explosives_synthesis_guidance", "real_lab_synthesis_guidance",
    "ab_initio_accuracy_claim", "toy_model_as_measurement_claim",
])
def test_gated_actions_hold(action):
    assert m.gate(action) == "HOLD"


def test_sandbox_proceeds():
    assert m.gate("run_simulation") == "PROCEED"


def test_argon4_seat():
    decider = m.brains_report()["decider"]
    assert decider["seat"] == "Gemini Argon 4 Extended Thinking"
    assert decider["can_override_hold"] is False
    assert decider["live_calls"] == 0


def test_claude_and_flow_seats():
    brains = m.brains_report()
    assert brains["advisor"]["can_override_hold"] is False
    assert brains["flow"]["rendered"] is False
    assert brains["flow"]["live"] == 0


def test_hold_demo_exit_code(capsys):
    assert m.main(["hold-demo"]) == 2
    assert "drug_discovery_claim" in capsys.readouterr().out


def test_checks_all_pass():
    checks = m.run_checks()
    assert all(ok for _, ok in checks)
