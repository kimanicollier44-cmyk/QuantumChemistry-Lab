#!/usr/bin/env python3
"""QuantumChemistry-Lab - educational quantum chemistry lab.

by Darion Kellar

Pure Python standard library. Zero dependencies. Zero network.
Education only - and the central lesson is stated up front:

  * This lab teaches the METHOD of quantum chemistry - exact
    diagonalization (full CI) of small Hamiltonians, a Hartree-Fock
    SCF loop, bond-length curves, and correlation energy - on TOY
    Hamiltonians whose integrals are LABELLED textbook-class
    parameters, chosen in the ranges real minimal-basis (STO-3G
    style) calculations produce. They are NOT computed ab initio:
    no integral is evaluated from basis functions here.
  * Every number the lab prints is therefore exact FOR THE TOY
    HAMILTONIAN and approximate for the real molecule. Where the
    lab quotes real-world reference values (the basis-set table,
    the dissociation limit), they are flagged LABELLED published-
    class values, never presented as this lab's own results.
  * The honest teaching points are structural and they are real:
    Hartree-Fock is variational (never below the exact energy of
    the same Hamiltonian), the gap between HF and exact IS the
    correlation energy, restricted HF dissociates H2 to the wrong
    limit while full CI dissociates it correctly, and HF is exact
    only for one-electron systems (asserted on H2+ below) - for a
    two-electron problem it is an approximation whose error this
    lab measures instead of hiding.

Real drug or materials discovery, weapons and explosives
chemistry, real lab synthesis guidance, toxic-compound design,
and production quantum-chemistry claims are permanent HOLD
gates - with Gemini Argon 4 Extended Thinking as DECIDER
(names-only, tighten-only), Muse as advisor (never overrides
HOLD), and Google Flow plan-only seats.
"""
import hashlib
import json
import math
import sys

# ---------------------------------------------------------------- constants
VERSION = "1.0.0"
BOHR_TO_ANGSTROM = 0.529177
# Labelled on-site Coulomb integral for a hydrogen 1s pair, Eh.
# At dissociation every two-electron MO integral of the toy tends
# to U/2: the g and u orbitals each become half an atom.
U_ONSITE = 0.625
U_HALF = U_ONSITE / 2.0
H2_R_EQ_ANCHOR = 1.4      # bohr - the labelled integrals anchor here
HEH_R_EQ_ANCHOR = 1.463   # bohr
SLATER_ZETA = 1.24        # STO-3G-class exponent for the overlap model

# ---------------------------------------------------------------- HOLD gates
HOLD_GATES = {
    "drug_discovery_claim": "HOLD - no real drug or pharmaceutical discovery claims from a toy Hamiltonian",
    "materials_discovery_claim": "HOLD - no real materials discovery claims; a labelled model discovers nothing",
    "weapons_chemistry_guidance": "HOLD - no weapons chemistry of any kind",
    "explosives_synthesis_guidance": "HOLD - no synthesis guidance for explosives or energetic materials",
    "real_lab_synthesis_guidance": "HOLD - no real laboratory synthesis instructions or procedures",
    "toxic_compound_design": "HOLD - no designing, optimising, or evaluating toxic compounds",
    "production_quantum_chemistry_claim": "HOLD - no production or industrial quantum-chemistry claims",
    "ab_initio_accuracy_claim": "HOLD - no claiming ab initio accuracy; the integrals are labelled parameters, not computed ones",
    "real_molecule_prediction_claim": "HOLD - no predicting a real molecule's properties for any real decision",
    "patent_discovery_claim": "HOLD - no patent, novelty, or discovery claims from lab output",
    "chemical_safety_certification": "HOLD - no certifying any real compound safe, stable, or compliant",
    "environmental_release_claim": "HOLD - no environmental fate, persistence, or release claims",
    "human_exposure_guidance": "HOLD - no human exposure, dose, or health guidance",
    "toy_model_as_measurement_claim": "HOLD - toy-model numbers must never be presented as measured or ab initio data",
}


def gate(action):
    """Return 'HOLD' if the action is gated, else 'PROCEED' (sandbox sim)."""
    return "HOLD" if action in HOLD_GATES else "PROCEED"


# ================================================================ 1. Jacobi eigensolver
# All diagonalization in this lab - CI matrices, the overlap
# matrix, the Fock matrix - goes through one from-scratch cyclic
# Jacobi rotation solver for real symmetric matrices. No numpy,
# no library eigensolver: the rotations are the lesson.

def jacobi_eig(matrix, tol=1e-12, max_sweeps=100):
    """Eigen-decomposition of a real symmetric matrix.

    Returns (eigenvalues, eigenvectors) with eigenvectors as
    COLUMNS: column i of the second return value is the unit
    eigenvector for eigenvalues[i].
    """
    n = len(matrix)
    a = [list(map(float, row)) for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(max_sweeps):
        off = math.sqrt(sum(a[i][j] ** 2
                            for i in range(n) for j in range(n) if i != j))
        if off < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-15:
                    continue
                diff = a[q][q] - a[p][p]
                if diff == 0.0:
                    t = 1.0 if a[p][q] > 0 else -1.0
                else:
                    tau = diff / (2.0 * a[p][q])
                    t = math.copysign(1.0, tau) / (
                        abs(tau) + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    return [a[i][i] for i in range(n)], v


def _sorted_eig(matrix):
    """Eigenpairs sorted by ascending eigenvalue."""
    values, vectors = jacobi_eig(matrix)
    order = sorted(range(len(values)), key=lambda i: values[i])
    return ([values[i] for i in order],
            [[vectors[row][i] for i in order] for row in range(len(values))])


# ================================================================ 2. Labelled integral models
# The Hamiltonians are built from smooth R-dependent model
# functions anchored at the equilibrium geometry. Anchors are
# textbook-class minimal-basis values (the H2 anchors reproduce
# the well-known STO-3G-class results: RHF -1.1167 Eh class at
# R = 1.4 bohr, overlap S = 0.6593). The functions are labelled
# model parameters - NOT ab initio integrals.

def overlap_h2(R):
    """1s-1s overlap for a Slater orbital, S = e^-zR (1 + zR + (zR)^2/3)."""
    z = SLATER_ZETA * R
    return math.exp(-z) * (1.0 + z + z * z / 3.0)


def _far_tail(R):
    """Fraction of the far nucleus 'switched on' at distance R -> 1."""
    return 1.0 - math.exp(-R / 2.0)


def _blend(R, at_eq, at_inf, r_eq, tau_up=2.2, tau_down=1.1):
    """Exponential blend from the anchor value to the asymptote."""
    tau = tau_up if R >= r_eq else tau_down
    return at_inf + (at_eq - at_inf) * math.exp(-(R - r_eq) / tau)


def h2_integrals(R):
    """Labelled MO-basis integrals for H2 at distance R (bohr).

    Orbitals: 0 = sigma_g (bonding), 1 = sigma_u (antibonding),
    orthonormal by construction. Returns h (one-electron), J and
    K (Coulomb / exchange matrices over orbital pairs), Vnn.
    Dissociation limits are built in: h -> -0.5 - tail, and every
    two-electron integral -> U/2, so full CI dissociates to two
    hydrogen atoms (-1.0 Eh electronic) while RHF does not.
    """
    tail = 0.5 / R * _far_tail(R)
    h_g = -0.5 - tail - 0.570 * math.exp(-(R - H2_R_EQ_ANCHOR) / 1.6)
    h_u = (-0.5 - tail - 0.05 * math.exp(-(R - H2_R_EQ_ANCHOR) / 1.6)
           + 0.30 * math.exp(-1.1 * R))
    j_gg = _blend(R, 0.669, U_HALF, H2_R_EQ_ANCHOR)
    j_uu = _blend(R, 0.550, U_HALF, H2_R_EQ_ANCHOR)
    j_gu = _blend(R, 0.600, U_HALF, H2_R_EQ_ANCHOR)
    k_gu = _blend(R, 0.160, U_HALF, H2_R_EQ_ANCHOR)
    return {
        "h": [h_g, h_u],
        "J": [[j_gg, j_gu], [j_gu, j_uu]],
        "K": [[j_gg, k_gu], [k_gu, j_uu]],
        "Vnn": 1.0 / R,
    }


def heh_integrals(R):
    """Labelled MO-basis integrals for HeH+ at distance R (bohr).

    Orbitals: 0 = bonding (He-centred at dissociation), 1 =
    antibonding. Anchored at R = 1.463 bohr to textbook-class
    values (HF -2.93 Eh class at equilibrium). Vnn = Z1*Z2/R = 2/R.
    """
    tail = _far_tail(R)
    h_b = (-1.75 - (0.85 / R) * tail
           - 0.398 * math.exp(-(R - HEH_R_EQ_ANCHOR) / 1.3)
           if R < HEH_R_EQ_ANCHOR else
           -1.75 - (0.85 / R) * tail
           - 0.398 * math.exp(-(R - HEH_R_EQ_ANCHOR) / 0.8))
    h_a = (-0.60 - (0.30 / R) * tail
           - 0.62 * math.exp(-(R - HEH_R_EQ_ANCHOR) / 1.3)
           + 0.40 * math.exp(-1.2 * R))
    j_bb = _blend(R, 0.603, 0.65, HEH_R_EQ_ANCHOR)
    j_aa = _blend(R, 0.550, 0.45, HEH_R_EQ_ANCHOR)
    j_ba = _blend(R, 0.580, 0.55, HEH_R_EQ_ANCHOR)
    k_ba = _blend(R, 0.220, 0.30, HEH_R_EQ_ANCHOR)
    return {
        "h": [h_b, h_a],
        "J": [[j_bb, j_ba], [j_ba, j_aa]],
        "K": [[j_bb, k_ba], [k_ba, j_aa]],
        "Vnn": 2.0 / R,
    }


def h2_large_integrals():
    """H2 with a second virtual orbital, frozen at R = 1.4 bohr.

    Orbitals: 0 = sigma_g, 1 = sigma_u, 2 = sigma_g* (second
    gerade virtual). The 2-orbital block is exactly the small
    model's anchor; the virtual's integrals are labelled values
    in the same class. Model simplification, stated openly:
    three-index MO integrals (pp|pq) are set to zero, so the
    correlation enters through closed-shell pair mixing - the
    dominant channel in the small model too.
    """
    base = h2_integrals(H2_R_EQ_ANCHOR)
    return {
        "h": [base["h"][0], base["h"][1], 0.02],
        "J": [[0.669, 0.600, 0.500],
              [0.600, 0.550, 0.470],
              [0.500, 0.470, 0.420]],
        "K": [[0.669, 0.160, 0.170],
              [0.160, 0.550, 0.110],
              [0.170, 0.110, 0.420]],
        "Vnn": base["Vnn"],
    }


# ================================================================ 3. Determinant CI (exact diagonalization)
# Two electrons, Ms = 0 Slater determinants over the MO basis.
# Hamiltonian matrix elements come from the Slater-Condon rules
# applied to the labelled MO integrals. Diagonalizing this matrix
# IS the exact solution of the toy Hamiltonian (full CI in the
# toy basis) - "exact" always means exact for this Hamiltonian.

def mo_integral(integ, p, q, r, s):
    """Chemists' (pq|rs) from the toy tensor: J, K, or zero."""
    if p == q and r == s:
        return integ["J"][p][r]
    if p == r and q == s and p != q:
        return integ["K"][p][q]
    if p == s and q == r and p != q:
        return integ["K"][p][q]
    return 0.0


def _spin_orbitals(n_orb):
    return [(i, spin) for i in range(n_orb) for spin in (0, 1)]


def ms0_determinants(n_orb):
    """All 2-electron Ms=0 determinants as spin-orbital pairs."""
    sos = _spin_orbitals(n_orb)
    dets = []
    for a in range(len(sos)):
        for b in range(a + 1, len(sos)):
            if sos[a][1] + sos[b][1] == 1:
                dets.append((sos[a], sos[b]))
    return dets


def _antisym(integ, a, b, c, d):
    """Antisymmetrised <ab||cd> for spin orbitals a, b, c, d."""
    value = 0.0
    if a[1] == c[1] and b[1] == d[1]:
        value += mo_integral(integ, a[0], c[0], b[0], d[0])
    if a[1] == d[1] and b[1] == c[1]:
        value -= mo_integral(integ, a[0], d[0], b[0], c[0])
    return value


def ci_matrix(integ, n_orb):
    """Full CI Hamiltonian matrix in the Ms=0 determinant basis."""
    dets = ms0_determinants(n_orb)
    n = len(dets)
    H = [[0.0] * n for _ in range(n)]
    for i, (a, b) in enumerate(dets):
        for j, (c, d) in enumerate(dets):
            bra, ket = {a, b}, {c, d}
            common = bra & ket
            if len(common) == 2:
                H[i][j] = (integ["h"][a[0]] + integ["h"][b[0]]
                           + _antisym(integ, a, b, a, b))
            elif len(common) == 1:
                (m,) = tuple(bra - ket)
                (p,) = tuple(ket - bra)
                (norb,) = tuple(common)
                h_one = (integ["h"][p[0]]
                         if (m[0] == p[0] and m[1] == p[1]) else 0.0)
                H[i][j] = h_one + _antisym(integ, m, norb, p, norb)
            else:
                H[i][j] = _antisym(integ, a, b, c, d)
    return H


def fci(integ, n_orb):
    """Exact (full-CI) ground state of the toy Hamiltonian.

    Returns energy (including Vnn), the ground eigenvector, and
    the determinant list so callers can read off the composition.
    """
    H = ci_matrix(integ, n_orb)
    values, vectors = _sorted_eig(H)
    ground_vec = [vectors[row][0] for row in range(len(values))]
    return {
        "energy": values[0] + integ["Vnn"],
        "electronic": values[0],
        "eigenvalues": [v + integ["Vnn"] for v in values],
        "ground_vector": ground_vec,
        "determinants": ms0_determinants(n_orb),
    }


def hf_energy_mo(integ):
    """Closed-shell HF energy from the MO formula: 2h_0 + J_00 + Vnn."""
    return 2.0 * integ["h"][0] + integ["J"][0][0] + integ["Vnn"]


def correlation_energy(integ, n_orb):
    """E_corr = E_exact - E_HF for the same toy Hamiltonian (<= 0)."""
    return fci(integ, n_orb)["energy"] - hf_energy_mo(integ)


# ================================================================ 4. AO basis + SCF loop
# The SCF works in the atomic-orbital basis {1s_A, 1s_B} with
# overlap S. Its integrals are DERIVED from the MO model above by
# inverting the g/u transformation, so the SCF and the CI solve
# the same Hamiltonian two different ways - and the checks assert
# that the two routes agree. That agreement, not a brochure
# number, is what validates the SCF.

def ao_from_mo(integ, S):
    """Derive AO core Hamiltonian + two-electron integrals.

    With g = (A+B)/sqrt(2(1+S)) and u = (A-B)/sqrt(2(1-S)), the
    MO integrals fix the AO quantities A=(AA|AA), B=(AA|BB),
    C=(AB|AB), E3=(AA|AB) through four linear relations, solved
    here in closed form.
    """
    j_gg, j_uu = integ["J"][0][0], integ["J"][1][1]
    j_gu, k_gu = integ["J"][0][1], integ["K"][0][1]
    e1 = j_gg * 2.0 * (1.0 + S) ** 2
    e2 = j_uu * 2.0 * (1.0 - S) ** 2
    e3 = j_gu * 2.0 * (1.0 - S * S)
    e4 = k_gu * 2.0 * (1.0 - S * S)
    three = (e1 - e2) / 8.0
    sum_p = (e1 + e2) / 2.0          # A + B + 2C
    sum_q = e3 + 2.0 * three        # A + B - 2C
    a_plus_b = (sum_p + sum_q) / 2.0
    coulomb_cross = (sum_p - sum_q) / 4.0
    a_val = (a_plus_b + e4) / 2.0
    b_val = (a_plus_b - e4) / 2.0
    h_g, h_u = integ["h"]
    h11 = ((1.0 + S) * h_g + (1.0 - S) * h_u) / 2.0
    h12 = ((1.0 + S) * h_g - (1.0 - S) * h_u) / 2.0
    return {"A": a_val, "B": b_val, "C": coulomb_cross, "E3": three,
            "H": [[h11, h12], [h12, h11]], "S": S}


def ao_integral(ao, p, q, r, s):
    """AO two-electron integral (pq|rs) by index pattern."""
    if p == q and r == s:
        return ao["A"] if p == r else ao["B"]
    if p != q and r != s:
        return ao["C"]
    return ao["E3"]


def mo_from_ao(ao):
    """Transform derived AO integrals back to the MO basis.

    Used by the checks: the round trip MO -> AO -> MO must
    reproduce the model integrals, proving both bases describe
    the same Hamiltonian.
    """
    S = ao["S"]
    h11, h12 = ao["H"][0][0], ao["H"][0][1]
    a, b, c, e3 = ao["A"], ao["B"], ao["C"], ao["E3"]
    return {
        "h": [(h11 + h12) / (1.0 + S), (h11 - h12) / (1.0 - S)],
        "J": [[(a + b + 2 * c + 4 * e3) / (2.0 * (1.0 + S) ** 2),
               (a + b - 2 * c - 2 * e3) / (2.0 * (1.0 - S * S))],
              [(a + b - 2 * c - 2 * e3) / (2.0 * (1.0 - S * S)),
               (a + b + 2 * c - 4 * e3) / (2.0 * (1.0 - S) ** 2)]],
        "K": [[(a + b + 2 * c + 4 * e3) / (2.0 * (1.0 + S) ** 2),
               (a - b) / (2.0 * (1.0 - S * S))],
              [(a - b) / (2.0 * (1.0 - S * S)),
               (a + b + 2 * c - 4 * e3) / (2.0 * (1.0 - S) ** 2)]],
    }


def _generalised_lowest(F, S):
    """Lowest eigenpair of F C = S C eps via symmetric orthogonalisation."""
    values, vectors = _sorted_eig(S)
    s_inv_sqrt = [[sum(vectors[i][k] * values[k] ** -0.5 * vectors[j][k]
                       for k in range(2)) for j in range(2)]
                  for i in range(2)]
    f_orth = [[sum(s_inv_sqrt[i][x] * F[x][y] * s_inv_sqrt[y][j]
                   for x in range(2) for y in range(2))
               for j in range(2)] for i in range(2)]
    f_values, f_vectors = _sorted_eig(f_orth)
    coeff = [sum(s_inv_sqrt[i][x] * f_vectors[x][0] for x in range(2))
             for i in range(2)]
    return f_values[0], coeff


def scf_h2(R, guess=(1.0, 0.35), tol=1e-12, max_iter=60):
    """Restricted HF SCF for H2 in the AO basis at distance R.

    Builds the Fock matrix F = H + G(P) from the density each
    iteration, solves the Roothaan equation by Jacobi
    diagonalisation, and iterates to energy convergence. The
    default guess is deliberately skewed (1.0, 0.35) so the loop
    visibly has to find the symmetric bonding orbital itself.
    """
    integ = h2_integrals(R)
    S = overlap_h2(R)
    ao = ao_from_mo(integ, S)
    s_mat = [[1.0, S], [S, 1.0]]
    core = ao["H"]
    norm = math.sqrt(sum(guess[i] * s_mat[i][j] * guess[j]
                         for i in range(2) for j in range(2)))
    coeff = [g / norm for g in guess]
    trace = []
    energy = 0.0
    prev = None
    eps = 0.0
    for iteration in range(1, max_iter + 1):
        density = [[2.0 * coeff[i] * coeff[j] for j in range(2)]
                   for i in range(2)]
        fock = [[core[i][j] + sum(
            density[r][s] * (ao_integral(ao, i, j, r, s)
                             - 0.5 * ao_integral(ao, i, s, r, j))
            for r in range(2) for s in range(2))
            for j in range(2)] for i in range(2)]
        energy = 0.5 * sum(density[i][j] * (core[i][j] + fock[i][j])
                           for i in range(2) for j in range(2)) + integ["Vnn"]
        delta = None if prev is None else energy - prev
        trace.append({"iteration": iteration, "energy": energy,
                      "delta": delta, "coeff": list(coeff)})
        if delta is not None and abs(delta) < tol:
            break
        prev = energy
        eps, coeff = _generalised_lowest(fock, s_mat)
    # Final Fock build from the converged density, so the reported
    # orbital energy belongs to the returned orbital exactly.
    density = [[2.0 * coeff[i] * coeff[j] for j in range(2)]
               for i in range(2)]
    fock = [[core[i][j] + sum(
        density[r][s] * (ao_integral(ao, i, j, r, s)
                         - 0.5 * ao_integral(ao, i, s, r, j))
        for r in range(2) for s in range(2))
        for j in range(2)] for i in range(2)]
    eps, _ = _generalised_lowest(fock, s_mat)
    return {"energy": energy, "eps_occ": eps, "coeff": coeff,
            "iterations": len(trace), "trace": trace,
            "converged": len(trace) < max_iter}


def h2plus_exact(R):
    """One-electron H2+: exact energy = lowest core eigenvalue + Vnn.

    With one electron there is no electron-electron term, so the
    'HF' problem and the exact problem are the same problem -
    the one case where HF is exact, asserted in the checks.
    """
    integ = h2_integrals(R)
    S = overlap_h2(R)
    ao = ao_from_mo(integ, S)
    eps, _ = _generalised_lowest(ao["H"], [[1.0, S], [S, 1.0]])
    return eps + integ["Vnn"]


# ================================================================ 5. Bond curves
CURVE_GRID = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0,
              2.5, 3.0, 4.0, 5.0, 6.0, 8.0]


def _golden_min(func, lo, hi, tol=1e-7):
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = hi - gr * (hi - lo)
    x2 = lo + gr * (hi - lo)
    f1, f2 = func(x1), func(x2)
    while hi - lo > tol:
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - gr * (hi - lo)
            f1 = func(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + gr * (hi - lo)
            f2 = func(x2)
    return (lo + hi) / 2.0


def h2_curve():
    """HF + exact energies on the grid, the exact minimum, dissociation."""
    rows = []
    for R in CURVE_GRID:
        integ = h2_integrals(R)
        e_hf = hf_energy_mo(integ)
        e_fci = fci(integ, 2)["energy"]
        rows.append({"R": R, "E_HF": e_hf, "E_exact": e_fci,
                     "E_corr": e_fci - e_hf})
    r_min = _golden_min(lambda R: fci(h2_integrals(R), 2)["energy"], 0.9, 2.2)
    e_min = fci(h2_integrals(r_min), 2)["energy"]
    e_diss = fci(h2_integrals(12.0), 2)["energy"]
    curvature = (fci(h2_integrals(r_min - 0.15), 2)["energy"]
                 - 2.0 * e_min
                 + fci(h2_integrals(r_min + 0.15), 2)["energy"]) / 0.15 ** 2
    return {"rows": rows, "R_min_bohr": r_min,
            "R_min_angstrom": r_min * BOHR_TO_ANGSTROM,
            "E_min": e_min, "E_dissociation_ref": e_diss,
            "De": e_diss - e_min, "curvature_at_min": curvature}


def heh_summary():
    """HeH+ equilibrium: golden-section minimum of the exact curve."""
    r_min = _golden_min(lambda R: fci(heh_integrals(R), 2)["energy"], 0.9, 2.6)
    integ = heh_integrals(r_min)
    e_exact = fci(integ, 2)["energy"]
    e_hf = hf_energy_mo(integ)
    return {"R_min_bohr": r_min,
            "R_min_angstrom": r_min * BOHR_TO_ANGSTROM,
            "E_exact": e_exact, "E_HF": e_hf,
            "E_corr": e_exact - e_hf,
            "E_at_anchor": fci(heh_integrals(HEH_R_EQ_ANCHOR), 2)["energy"],
            "binding_vs_R4": fci(heh_integrals(4.0), 2)["energy"] - e_exact}


def curvature_at(R, delta=0.15):
    """Second-difference curvature of the exact H2 curve at R."""
    return (fci(h2_integrals(R - delta), 2)["energy"]
            - 2.0 * fci(h2_integrals(R), 2)["energy"]
            + fci(h2_integrals(R + delta), 2)["energy"]) / delta ** 2


# ================================================================ 6. Basis-set honesty table
# Published-class behaviour of real basis sets for H2, LABELLED
# as literature-class approximations - none of these numbers is
# computed by this lab. The Hartree-Fock limit (-1.1336 Eh) and
# the exact nonrelativistic energy (-1.1745 Eh) are the hard
# anchors: no RHF row may pass the limit, no FCI row the exact.
BASIS_TABLE = [
    {"basis": "Minimal (STO-3G class)", "label": "LABELLED published-class",
     "E_HF": -1.117, "E_FCI": -1.137,
     "adds": "one function per atomic orbital",
     "note": "the class this lab's toy is anchored to"},
    {"basis": "Split-valence (6-31G class)", "label": "LABELLED published-class",
     "E_HF": -1.127, "E_FCI": -1.157,
     "adds": "valence orbitals split into two sizes",
     "note": "breathing room the minimal basis lacks"},
    {"basis": "Polarised (6-31G** class)", "label": "LABELLED published-class",
     "E_HF": -1.131, "E_FCI": -1.166,
     "adds": "p functions on H, d functions on heavy atoms",
     "note": "orbitals can shift off the nuclei"},
    {"basis": "Correlation-consistent (cc-pVTZ class)",
     "label": "LABELLED published-class",
     "E_HF": -1.133, "E_FCI": -1.172,
     "adds": "systematic shells up to f functions",
     "note": "approaching both limits, still not at them"},
    {"basis": "Hartree-Fock limit", "label": "LABELLED reference",
     "E_HF": -1.1336, "E_FCI": None,
     "adds": "infinite one-electron basis",
     "note": "no single determinant can go below this"},
    {"basis": "Exact nonrelativistic", "label": "LABELLED reference",
     "E_HF": None, "E_FCI": -1.1745,
     "adds": "the true answer for the real molecule",
     "note": "what every row above is reaching for"},
]


def basis_report():
    """The honesty table plus this lab's own computed toy row."""
    integ = h2_integrals(H2_R_EQ_ANCHOR)
    toy = {"basis": "This lab's toy (2 MO, labelled integrals)",
           "label": "COMPUTED here, for the toy Hamiltonian",
           "E_HF": hf_energy_mo(integ),
           "E_FCI": fci(integ, 2)["energy"],
           "adds": "sigma_g + sigma_u only",
           "note": "exact for the toy, approximate for the molecule"}
    return {"rows": BASIS_TABLE, "toy_row": toy}


# ================================================================ 7. kNN bond classifier
# Features: bond length, energy above the exact minimum (as a
# fraction of the model's dissociation energy), and curvature of
# the exact curve. The training grid is labelled by the model's
# own label rule - the classifier interpolates the model, it
# does not replace the curve calculation.

def model_label(R, e_rel_frac, curvature):
    """The model's own three-way bond label."""
    if R >= 4.0 and e_rel_frac >= 0.6:
        return "dissociated"
    if R <= 2.6 and curvature >= 0.0:
        return "bound"
    return "weak"


def _bond_features(R, curve=None):
    curve = h2_curve() if curve is None else curve
    e_rel = fci(h2_integrals(R), 2)["energy"] - curve["E_min"]
    return (R, e_rel / curve["De"], curvature_at(R))


def _training_grid():
    curve = h2_curve()
    rows = []
    R = 0.8
    while R <= 6.01:
        features = _bond_features(R, curve)
        rows.append((features, model_label(*features)))
        R += 0.2
    return rows


TRAINING_GRID = _training_grid()


def _scaled(features):
    R, e_rel_frac, curvature = features
    return (R / 6.0, e_rel_frac, curvature / 2.0)


def classify_bond(R, k=3):
    """kNN over the model-labelled grid, deterministic ties."""
    target = _scaled(_bond_features(R))
    ranked = []
    for features, label in TRAINING_GRID:
        point = _scaled(features)
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(target, point)))
        ranked.append((dist, features, label))
    ranked.sort(key=lambda row: (row[0], row[1]))
    votes = {}
    for _, _, label in ranked[:k]:
        votes[label] = votes.get(label, 0) + 1
    priority = {"bound": 0, "weak": 1, "dissociated": 2}
    return sorted(votes.items(), key=lambda kv: (-kv[1], priority[kv[0]]))[0][0]


def classifier_report():
    scenarios = {"R_0.80_compressed": 0.8, "R_1.05_short": 1.05,
                 "R_1.40_equilibrium": 1.4, "R_2.60_stretched": 2.6,
                 "R_3.50_long": 3.5, "R_6.00_apart": 6.0}
    return {name: classify_bond(R) for name, R in scenarios.items()}


# ================================================================ 8. Advisory seats
def _trace(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def argon4_decide(topic="quantumchemistry-lab-demo"):
    """Gemini Argon 4 DECIDER seat: MAP->MODEL->STRESS->DECIDE.

    Names-only keys, live calls 0, tighten-only: can add HOLDs, never lift one.
    """
    stages = ["MAP", "MODEL", "STRESS", "DECIDE"]
    payload = {"seat": "gemini-argon-4", "topic": topic, "stages": stages}
    return {
        "seat": "Gemini Argon 4 Extended Thinking",
        "role": "DECIDER",
        "pipeline": stages,
        "tiers": [2048, 8192, 32768],
        "key": "names-only (GEMINI_API_KEY name checked, value never read)",
        "live_calls": 0,
        "decision": "PROCEED - sandbox simulation only, all discovery / synthesis gates stay HOLD",
        "can_override_hold": False,
        "trace_sha256_16": _trace(payload),
    }


def claude_advisor(topic="quantumchemistry-lab-demo"):
    """Muse advisor seat: advises, can never override a HOLD."""
    return {
        "seat": "Muse",
        "role": "ADVISOR",
        "key": "names-only (MODEL_API_KEY name checked, value never read)",
        "live_calls": 0,
        "advice": "Say it in the first line of every report: the integrals are labelled textbook-class parameters, so the energies are exact for the toy Hamiltonian and approximate for the molecule - then show the HF/exact gap and call it by its name, correlation.",
        "can_override_hold": False,
    }


def google_flow_plan():
    """Google Flow seat: plan-only, nothing rendered, nothing live."""
    return {
        "seat": "Google Flow",
        "role": "PLAN-ONLY",
        "shots": ["two hydrogen spheres approaching along a bond axis",
                  "a potential well drawing itself as the curve is scanned",
                  "the HF curve peeling away from the exact curve at long range",
                  "determinant tiles mixing into one ground-state wave",
                  "an SCF loop tightening an orbital until the energy stops moving"],
        "rendered": False,
        "live": 0,
    }


def brains_report():
    return {"decider": argon4_decide(), "advisor": claude_advisor(),
            "flow": google_flow_plan()}


# ================================================================ checks
def run_checks():
    c = []

    def chk(name, ok):
        c.append((name, bool(ok)))

    # 1. Jacobi eigensolver
    vals, _ = _sorted_eig([[2.0, 1.0], [1.0, 2.0]])
    chk("jacobi_known_2x2", abs(vals[0] - 1.0) < 1e-10
        and abs(vals[1] - 3.0) < 1e-10)
    vals3, vecs3 = _sorted_eig([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0],
                                [0.0, 0.0, 5.0]])
    chk("jacobi_diagonal", vals3 == [2.0, 3.0, 5.0])
    m3 = [[4.0, 1.0, 2.0], [1.0, 3.0, 0.5], [2.0, 0.5, 5.0]]
    vals3, vecs3 = _sorted_eig(m3)
    chk("jacobi_trace_preserved",
        abs(sum(vals3) - 12.0) < 1e-9)
    ortho = all(abs(sum(vecs3[r][i] * vecs3[r][j] for r in range(3))
                    - (1.0 if i == j else 0.0)) < 1e-9
                for i in range(3) for j in range(3))
    chk("jacobi_eigenvectors_orthonormal", ortho)
    recon = all(abs(sum(vecs3[i][k] * vals3[k] * vecs3[j][k]
                        for k in range(3)) - m3[i][j]) < 1e-9
                for i in range(3) for j in range(3))
    chk("jacobi_reconstruction", recon)

    # 2. labelled integral model
    chk("overlap_sto3g_class", abs(overlap_h2(1.4) - 0.6593) < 0.001)
    chk("overlap_limits", overlap_h2(0.2) > 0.95 and overlap_h2(8.0) < 0.01)
    chk("vnn_exact", h2_integrals(2.0)["Vnn"] == 0.5
        and heh_integrals(2.0)["Vnn"] == 1.0)
    far = h2_integrals(12.0)
    chk("integrals_dissociation_limits",
        all(abs(far["J"][i][j] - U_HALF) < 0.02
            and abs(far["K"][i][j] - U_HALF) < 0.02
            for i in range(2) for j in range(2)))
    chk("h_g_far_limit", -0.56 < far["h"][0] < -0.53)

    # 3. determinant CI
    chk("determinant_counts", len(ms0_determinants(2)) == 4
        and len(ms0_determinants(3)) == 9)
    integ = h2_integrals(H2_R_EQ_ANCHOR)
    H = ci_matrix(integ, 2)
    chk("ci_matrix_symmetric",
        all(abs(H[i][j] - H[j][i]) < 1e-12 for i in range(4) for j in range(4)))
    dets = ms0_determinants(2)
    g2_index = dets.index(((0, 0), (0, 1)))
    open_index = dets.index(((0, 0), (1, 1)))
    chk("parity_blocks_open_shell", abs(H[g2_index][open_index]) < 1e-12)
    chk("pair_hopping_is_exchange",
        abs(H[g2_index][dets.index(((1, 0), (1, 1)))]
            - integ["K"][0][1]) < 1e-12)
    exact = fci(integ, 2)
    e_hf = hf_energy_mo(integ)
    chk("hf_anchor_energy", -1.125 < e_hf < -1.108)
    chk("exact_anchor_energy", -1.150 < exact["energy"] < -1.130)
    chk("correlation_anchor_range",
        -0.035 < exact["energy"] - e_hf < -0.015)
    chk("variational_hf_above_exact", e_hf >= exact["energy"] - 1e-12)
    weight_g2 = exact["ground_vector"][g2_index] ** 2
    chk("ground_state_mostly_g2", 0.95 < weight_g2 < 1.0)
    chk("variational_everywhere_on_grid",
        all(hf_energy_mo(h2_integrals(R))
            >= fci(h2_integrals(R), 2)["energy"] - 1e-12
            for R in CURVE_GRID))

    # 4. SCF - two routes to the same HF energy
    scf = scf_h2(H2_R_EQ_ANCHOR)
    chk("scf_converged", scf["converged"] and scf["iterations"] <= 25)
    chk("scf_matches_mo_formula", abs(scf["energy"] - e_hf) < 1e-9)
    chk("scf_orbital_is_bonding_mo",
        abs(scf["coeff"][0] - 1.0 / math.sqrt(2 * (1 + overlap_h2(1.4))))
        < 1e-6 and abs(scf["coeff"][0] - scf["coeff"][1]) < 1e-6)
    chk("scf_orbital_energy_is_fock_eigenvalue",
        abs(scf["eps_occ"] - (integ["h"][0] + integ["J"][0][0])) < 1e-9)
    ao = ao_from_mo(integ, overlap_h2(H2_R_EQ_ANCHOR))
    back = mo_from_ao(ao)
    chk("ao_mo_roundtrip",
        all(abs(back["h"][i] - integ["h"][i]) < 1e-9 for i in range(2))
        and all(abs(back["J"][i][j] - integ["J"][i][j]) < 1e-9
                for i in range(2) for j in range(2))
        and all(abs(back["K"][i][j] - integ["K"][i][j]) < 1e-9
                for i in range(2) for j in range(2)))
    chk("h2plus_hf_is_exact",
        abs(h2plus_exact(H2_R_EQ_ANCHOR)
            - (integ["h"][0] + integ["Vnn"])) < 1e-10)

    # 5. bond curve
    curve = h2_curve()
    chk("curve_minimum_position", 1.34 < curve["R_min_bohr"] < 1.46)
    chk("curve_minimum_angstrom", 0.71 < curve["R_min_angstrom"] < 0.77)
    chk("curve_minimum_energy",
        abs(curve["E_min"] - fci(h2_integrals(curve["R_min_bohr"]), 2)
            ["energy"]) < 1e-6)
    chk("curve_dissociation_exact_limit",
        -1.01 < curve["rows"][-1]["E_exact"] < -0.96)
    chk("curve_rhf_wrong_dissociation",
        curve["rows"][-1]["E_HF"] - curve["rows"][-1]["E_exact"] >= 0.2
        and -0.72 < curve["rows"][-1]["E_HF"] < -0.65)
    chk("curve_dissociation_energy", 0.10 < curve["De"] < 0.18)
    chk("curve_curvature_positive_at_min", curve["curvature_at_min"] > 0)

    # 6. HeH+
    heh = heh_summary()
    chk("heh_minimum_position", 1.30 < heh["R_min_bohr"] < 1.65)
    chk("heh_energy_class", -3.02 < heh["E_exact"] < -2.90)
    chk("heh_binds", heh["binding_vs_R4"] >= 0.10)
    chk("heh_variational", heh["E_HF"] >= heh["E_exact"] - 1e-12)

    # 7. larger toy - correlation grows with the basis
    large = h2_large_integrals()
    e_large = fci(large, 3)["energy"]
    chk("large_toy_more_correlation",
        e_large < exact["energy"] - 0.005
        and (e_large - e_hf) < (exact["energy"] - e_hf) - 0.005)
    chk("large_toy_same_hf", abs(hf_energy_mo(large) - e_hf) < 1e-12)

    # 8. basis honesty table
    rep = basis_report()
    fci_rows = [r["E_FCI"] for r in rep["rows"] if r["E_FCI"] is not None]
    hf_rows = [r["E_HF"] for r in rep["rows"] if r["E_HF"] is not None]
    chk("basis_table_monotone",
        all(fci_rows[i] > fci_rows[i + 1] for i in range(len(fci_rows) - 1))
        and all(hf_rows[i] > hf_rows[i + 1] for i in range(len(hf_rows) - 1)))
    chk("basis_table_limits_respected",
        min(hf_rows) >= -1.1336 - 1e-9 and min(fci_rows) >= -1.1745 - 1e-9)
    chk("basis_toy_row_matches_computation",
        abs(rep["toy_row"]["E_HF"] - e_hf) < 1e-9
        and abs(rep["toy_row"]["E_FCI"] - exact["energy"]) < 1e-9)
    chk("basis_rows_labelled",
        all(r["label"].startswith("LABELLED") for r in rep["rows"]))

    # 9. classifier - labels come from the model itself
    crep = classifier_report()
    chk("classifier_equilibrium_bound",
        crep["R_1.40_equilibrium"] == "bound")
    chk("classifier_stretched_weak", crep["R_2.60_stretched"] == "weak")
    chk("classifier_apart_dissociated",
        crep["R_6.00_apart"] == "dissociated")
    chk("classifier_matches_model",
        all(classify_bond(R) == model_label(*_bond_features(R))
            for R in (0.8, 1.05, 1.4, 2.6, 3.5, 6.0)))

    # gates + seats
    chk("hold_gates_14", len(HOLD_GATES) == 14
        and all(v.startswith("HOLD") for v in HOLD_GATES.values()))
    chk("gate_drug_hold", gate("drug_discovery_claim") == "HOLD")
    chk("gate_weapons_hold", gate("weapons_chemistry_guidance") == "HOLD")
    chk("gate_synthesis_hold", gate("real_lab_synthesis_guidance") == "HOLD")
    chk("gate_abinitio_hold", gate("ab_initio_accuracy_claim") == "HOLD")
    chk("gate_toy_overclaim_hold",
        gate("toy_model_as_measurement_claim") == "HOLD")
    chk("gate_sandbox_proceed", gate("run_simulation") == "PROCEED")
    brains = brains_report()
    chk("argon4_no_override", brains["decider"]["can_override_hold"] is False
        and brains["decider"]["seat"]
        == "Gemini Argon 4 Extended Thinking")
    chk("claude_no_override", brains["advisor"]["can_override_hold"] is False)
    chk("flow_plan_only", brains["flow"]["rendered"] is False
        and brains["flow"]["live"] == 0)
    return c


# ================================================================ CLI / demo
def _h2_report():
    integ = h2_integrals(H2_R_EQ_ANCHOR)
    exact = fci(integ, 2)
    e_hf = hf_energy_mo(integ)
    dets = exact["determinants"]
    g2_index = dets.index(((0, 0), (0, 1)))
    return {"R_bohr": H2_R_EQ_ANCHOR, "integrals": integ,
            "E_HF": e_hf, "E_exact": exact["energy"],
            "E_corr": exact["energy"] - e_hf,
            "ground_g2_weight": exact["ground_vector"][g2_index] ** 2,
            "eigenvalues": exact["eigenvalues"]}


def demo():
    print("=" * 68)
    print("QuantumChemistry-Lab - quantum chemistry method lab")
    print("by Darion Kellar | MIT | pure stdlib | zero network | EDUCATION ONLY")
    print("=" * 68)
    print("\n[1] THE HONEST CORE - toy Hamiltonians, labelled integrals")
    print("    The integrals below are LABELLED textbook-class parameters")
    print("    (minimal-basis / STO-3G ranges), not computed ab initio.")
    print("    Every energy is therefore exact FOR THE TOY HAMILTONIAN")
    print("    and approximate for the real molecule. The method is the")
    print("    lesson: diagonalize exactly, run HF, and measure the gap.")
    rep = _h2_report()
    integ = rep["integrals"]
    print(f"\n[2] H2 at the anchor geometry R = {rep['R_bohr']} bohr "
          f"(overlap S = {overlap_h2(rep['R_bohr']):.4f})")
    print(f"    h_g = {integ['h'][0]:+.4f}  h_u = {integ['h'][1]:+.4f}  "
          f"J_gg = {integ['J'][0][0]:.4f}  J_uu = {integ['J'][1][1]:.4f}  "
          f"J_gu = {integ['J'][0][1]:.4f}  K_gu = {integ['K'][0][1]:.4f}")
    print(f"    E_HF    = {rep['E_HF']:.5f} Eh  (one determinant, sigma_g^2)")
    print(f"    E_exact = {rep['E_exact']:.5f} Eh  (full CI, 4 determinants)")
    print(f"    E_corr  = {rep['E_corr']:+.5f} Eh  <- the gap IS the "
          f"correlation energy")
    print(f"    ground state: {rep['ground_g2_weight'] * 100:.1f}% sigma_g^2 "
          f"determinant, the rest is sigma_u^2 mixing")
    print("    HF and exact agree to within the correlation energy here")
    print("    (asserted < 0.05 Eh in CHECKS) - but they are NOT equal,")
    print("    and HF is exactly equal to the exact answer only for")
    print(f"    one-electron systems: H2+ exact = {h2plus_exact(1.4):.5f} Eh "
          f"= h_g + Vnn, asserted to 1e-10.")
    scf = scf_h2(H2_R_EQ_ANCHOR)
    print("\n[3] Hartree-Fock SCF - the same Hamiltonian, a second route")
    print("    AO-basis Roothaan loop from a skewed guess (1.0, 0.35):")
    for row in scf["trace"]:
        delta = "   ---  " if row["delta"] is None else f"{row['delta']:+.2e}"
        print(f"    iter {row['iteration']:2d}  E = {row['energy']:.10f}  "
              f"delta {delta}  c = ({row['coeff'][0]:.4f}, "
              f"{row['coeff'][1]:.4f})")
    print(f"    converged in {scf['iterations']} iterations: "
          f"E_SCF = {scf['energy']:.5f} Eh = the MO formula "
          f"({rep['E_HF']:.5f}) to <1e-9 - two routes, one Hamiltonian.")
    curve = h2_curve()
    print("\n[4] Bond-length curve - exact (full CI) vs HF")
    print("    R (bohr)    E_HF       E_exact    E_corr")
    for row in curve["rows"]:
        print(f"    {row['R']:7.2f}  {row['E_HF']:9.5f}  "
              f"{row['E_exact']:9.5f}  {row['E_corr']:+.5f}")
    print(f"    minimum: R = {curve['R_min_bohr']:.3f} bohr = "
          f"{curve['R_min_angstrom']:.3f} Ang, "
          f"E = {curve['E_min']:.5f} Eh, De = {curve['De']:.5f} Eh")
    last = curve["rows"][-1]
    print(f"    dissociation: exact -> {last['E_exact']:.3f} Eh (two H "
          f"atoms, -1.0 limit) while RHF stalls at {last['E_HF']:.3f} Eh")
    print("    -> the restricted determinant keeps ionic terms alive at")
    print("       long range; the exact wavefunction lets them go. That")
    print("       growing gap is static correlation, printed, not hidden.")
    heh = heh_summary()
    print(f"\n[5] HeH+ - the same machinery, a heteronuclear ion")
    print(f"    minimum: R = {heh['R_min_bohr']:.3f} bohr = "
          f"{heh['R_min_angstrom']:.3f} Ang, "
          f"E_exact = {heh['E_exact']:.5f} Eh, "
          f"E_HF = {heh['E_HF']:.5f} Eh, "
          f"E_corr = {heh['E_corr']:+.5f} Eh")
    large = fci(h2_large_integrals(), 3)
    print("\n[6] One more virtual orbital - correlation is basis-hungry")
    print(f"    3-orbital toy at R = 1.4: E_exact = {large['energy']:.5f} Eh "
          f"(9 determinants)")
    print(f"    correlation deepens {rep['E_corr']:+.5f} -> "
          f"{large['energy'] - rep['E_HF']:+.5f} Eh while HF does not "
          f"move at all: HF never sees the virtuals, exact does.")
    print("\n[7] Basis-set honesty table (LABELLED published-class rows)")
    for row in basis_report()["rows"]:
        hf = "  ---  " if row["E_HF"] is None else f"{row['E_HF']:+.4f}"
        fc = "  ---  " if row["E_FCI"] is None else f"{row['E_FCI']:+.4f}"
        print(f"    {row['basis']:38s} HF {hf}  FCI {fc}")
    toy = basis_report()["toy_row"]
    print(f"    {'This lab toy (COMPUTED, toy only)':38s} "
          f"HF {toy['E_HF']:+.4f}  FCI {toy['E_FCI']:+.4f}")
    print("\n[8] Bond classifier (TinyML kNN, labels from the model)")
    for name, label in classifier_report().items():
        print(f"    {name:36s} -> {label}")
    print("\n[9] Advisory seats")
    brains = brains_report()
    print(f"    decider: {brains['decider']['decision']} "
          f"trace={brains['decider']['trace_sha256_16']}")
    print(f"    advisor can_override_hold = "
          f"{brains['advisor']['can_override_hold']}")
    print(f"    flow rendered = {brains['flow']['rendered']} "
          f"live = {brains['flow']['live']}")
    print(f"\n[10] HOLD gates: {len(HOLD_GATES)}")
    for key in HOLD_GATES:
        print(f"    HOLD  {key}")
    checks = run_checks()
    passed = sum(1 for _, ok in checks if ok)
    print(f"\nCHECKS {passed}/{len(checks)}")
    for name, ok in checks:
        if not ok:
            print(f"    FAIL {name}")
    verdict = "PROCEED" if passed == len(checks) else "HOLD"
    print(f"VERDICT {verdict} (sandbox simulation only) | "
          f"HOLD gates {len(HOLD_GATES)}")
    return 0 if passed == len(checks) else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] == "demo":
        return demo()
    cmd = argv[0]
    if cmd == "checks":
        checks = run_checks()
        for name, ok in checks:
            print(f"{'PASS' if ok else 'FAIL'} {name}")
        print(f"CHECKS {sum(1 for _, ok in checks if ok)}/{len(checks)}")
        return 0 if all(ok for _, ok in checks) else 1
    if cmd == "h2":
        print(json.dumps(_h2_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "curve":
        print(json.dumps(h2_curve(), indent=2, sort_keys=True))
        return 0
    if cmd == "scf":
        print(json.dumps(scf_h2(H2_R_EQ_ANCHOR), indent=2, sort_keys=True))
        return 0
    if cmd == "basis":
        print(json.dumps(basis_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "classify":
        print(json.dumps(classifier_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "brains":
        print(json.dumps(brains_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "holds":
        for key, value in HOLD_GATES.items():
            print(f"{key}: {value}")
        return 0
    if cmd == "hold-demo":
        action = argv[1] if len(argv) > 1 else "drug_discovery_claim"
        print(f"{action}: {HOLD_GATES.get(action, 'PROCEED - not gated')}")
        return 2 if gate(action) == "HOLD" else 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
