#!/usr/bin/env python3
"""
Q14 Temperature — tehnika: Kvantna temperatura preko aux superpozicije
(čisto kvantno, bez klasičnog softmax-a i bez hibrida).

Koncept:
  - „Temperatura“ T ∈ [0, 1] kontroliše balans između „sharp“ (CSV-signal)
    i „uniform“ („kreativno“) režima preko superpozicije sa 1 aux qubit-om.
  - Aux qubit: Ry(2α)|0⟩ = cos(α)|0⟩ + sin(α)|1⟩,  α = π·T/2.
    · T = 0 → aux = |0⟩ → čisto sharp (niska T, precizno).
    · T = 0.5 → aux = (|0⟩+|1⟩)/√2 → 50/50 mix.
    · T = 1 → aux = |1⟩ → čisto uniform (visoka T, raznolikost).

Kolo (nq + 1 qubit-a):
  1) Ry(2α) na aux.
  2) Kontrolisani StatePreparation(|ψ_sharp⟩) na state registar kad aux = 0
     (|ψ_sharp⟩ = amplitude-encoding freq_vector-a CELOG CSV-a).
  3) Kontrolisani Hadamard (CH) na svakom state qubit-u kad aux = 1
     (pripremi uniformno stanje iz |0⟩^nq).

Marginala nad state registrom:
  p[k] = cos²(α)·|ψ_sharp[k]|² + sin²(α)·(1/2^nq)
  → bias_39 → NEXT (TOP-7).

Grid i izbor T:
  - Grid se vodi SAMO nad nq, po meri cos(bias_39 pri T=0, freq_csv).
    (Razlog: za skoro-uniformni freq_csv, uniform bias bi lažno „pobeđivao“ u
     cos-meri; zato T nije deo grid-pretrage.)
  - Glavna predikcija koristi T_MAIN = 0.0 (niska T = precizno/pouzdano),
    što je kanonsko značenje Temperature = 0 u LLM semantici.
  - Demonstracija efekta: za odabran nq štampaju se NEXT i cos-skorovi za
    niz vrednosti T_DEMO ∈ {0.0, 0.3, 0.5, 0.7, 1.0}.

Sve deterministički: seed=39; amp_sharp iz CELOG CSV-a.

Okruženje: Python 3.11.13, qiskit 1.4.4, qiskit-machine-learning 0.8.3, macOS M1 (vidi README.md).
"""

from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector

# =========================
# Seed
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass

# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/data/loto7hh_4600_k31.csv")
N_NUMBERS = 7
N_MAX = 39

GRID_NQ = (5, 6)
T_MAIN = 0.0
T_DEMO = (0.0, 0.3, 0.5, 0.7, 1.0)


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def freq_vector(H: np.ndarray) -> np.ndarray:
    c = np.zeros(N_MAX, dtype=np.float64)
    for v in H.ravel():
        if 1 <= v <= N_MAX:
            c[int(v) - 1] += 1.0
    return c


def amp_from_freq(f: np.ndarray, nq: int) -> np.ndarray:
    dim = 2 ** nq
    edges = np.linspace(0, N_MAX, dim + 1, dtype=int)
    amp = np.array(
        [float(f[edges[i] : edges[i + 1]].mean()) if edges[i + 1] > edges[i] else 0.0 for i in range(dim)],
        dtype=np.float64,
    )
    amp = np.maximum(amp, 0.0)
    n2 = float(np.linalg.norm(amp))
    if n2 < 1e-18:
        amp = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    else:
        amp = amp / n2
    return amp


# =========================
# Temperature kolo — aux superpozicija sharp/uniform
# =========================
def build_temperature_state(H: np.ndarray, nq: int, T: float) -> Statevector:
    """|Ψ⟩ = cos(α)|0⟩_aux|ψ_sharp⟩_state + sin(α)|1⟩_aux|uniform⟩_state, α = π·T/2."""
    amp_sharp = amp_from_freq(freq_vector(H), nq)

    state = QuantumRegister(nq, name="s")
    aux = QuantumRegister(1, name="a")
    qc = QuantumCircuit(state, aux)

    alpha = float(np.pi * T / 2.0)
    qc.ry(2.0 * alpha, aux[0])

    sp = StatePreparation(amp_sharp.tolist())
    sp_ctrl = sp.control(num_ctrl_qubits=1, ctrl_state=0)
    qc.append(sp_ctrl, [aux[0]] + list(state))

    for k in range(nq):
        qc.ch(aux[0], state[k])

    return Statevector(qc)


def temperature_state_probs(H: np.ndarray, nq: int, T: float) -> np.ndarray:
    sv = build_temperature_state(H, nq, T)
    p = np.abs(sv.data) ** 2

    # Qiskit little-endian: state registar (dodat prvi) → niži bitovi,
    # aux → viši bit. reshape(2, 2^nq) daje [aux][state].
    dim_s = 2 ** nq
    mat = p.reshape(2, dim_s)
    p_s = mat.sum(axis=0)
    s = float(p_s.sum())
    return p_s / s if s > 0 else p_s


# =========================
# Readout
# =========================
def bias_39(probs: np.ndarray, n_max: int = N_MAX) -> np.ndarray:
    b = np.zeros(n_max, dtype=np.float64)
    for idx, p in enumerate(probs):
        b[idx % n_max] += float(p)
    s = float(b.sum())
    return b / s if s > 0 else b


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pick_next_combination(probs: np.ndarray, k: int = N_NUMBERS, n_max: int = N_MAX) -> Tuple[int, ...]:
    b = bias_39(probs, n_max)
    order = np.argsort(-b, kind="stable")
    return tuple(sorted(int(o + 1) for o in order[:k]))


# =========================
# Determ. grid-optimizacija SAMO nad nq (T = 0 baseline)
# =========================
def optimize_nq(H: np.ndarray):
    f_csv = freq_vector(H)
    s = float(f_csv.sum())
    f_csv_n = f_csv / s if s > 0 else np.ones(N_MAX) / N_MAX
    best = None
    for nq in GRID_NQ:
        try:
            p = temperature_state_probs(H, nq, 0.0)
            b = bias_39(p)
            score = cosine(b, f_csv_n)
        except Exception:
            continue
        key = (score, -nq)
        if best is None or key > best[0]:
            best = (key, dict(nq=nq, score=float(score)))
    return best[1] if best else None


def main() -> int:
    H = load_rows(CSV_PATH)
    if H.shape[0] < 1:
        print("premalo redova")
        return 1

    print("Q14 Temperature (aux superpozicija sharp/uniform): CSV:", CSV_PATH)
    print("redova:", H.shape[0], "| seed:", SEED)

    best = optimize_nq(H)
    if best is None:
        print("grid optimizacija nije uspela")
        return 2
    print(
        "BEST nq:", best["nq"],
        "| cos(bias@T=0, freq_csv):", round(float(best["score"]), 6),
    )

    nq_best = int(best["nq"])
    f_csv = freq_vector(H)
    s = float(f_csv.sum())
    f_csv_n = f_csv / s if s > 0 else np.ones(N_MAX) / N_MAX

    print("--- demonstracija efekta temperature (isti nq, različito T) ---")
    for T in T_DEMO:
        p_T = temperature_state_probs(H, nq_best, float(T))
        pred_T = pick_next_combination(p_T)
        cos_T = cosine(bias_39(p_T), f_csv_n)
        print(f"T={T:.2f}  cos(bias, freq_csv)={cos_T:.6f}  NEXT={pred_T}")

    p_main = temperature_state_probs(H, nq_best, T_MAIN)
    pred_main = pick_next_combination(p_main)
    print("--- glavna predikcija ---")
    print("T_MAIN=", T_MAIN, "| nq=", nq_best)
    print("predikcija NEXT:", pred_main)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Q14 Temperature (aux superpozicija sharp/uniform): CSV: /data/loto7hh_4600_k31.csv
redova: 4600 | seed: 39
BEST nq: 5 | cos(bias@T=0, freq_csv): 0.900351
--- demonstracija efekta temperature (isti nq, različito T) ---
T=0.00  cos(bias, freq_csv)=0.900351  NEXT=(7, 19, 22, 24, 27, 28, 31)
T=0.30  cos(bias, freq_csv)=0.901082  NEXT=(7, 19, 22, 24, 27, 28, 31)
T=0.50  cos(bias, freq_csv)=0.901783  NEXT=(7, 19, 22, 24, 27, 28, 31)
T=0.70  cos(bias, freq_csv)=0.902079  NEXT=(7, 19, 22, 24, 27, 28, 31)
T=1.00  cos(bias, freq_csv)=0.902043  NEXT=(1, 2, 3, 4, 5, 6, 7)
--- glavna predikcija ---
T_MAIN= 0.0 | nq= 5
predikcija NEXT: (7, 19, x, y, z, 28, 31)
"""



"""
Q14_Temperature_aux.py — tehnika: Kvantna temperatura preko aux superpozicije.

Kolo (nq + 1):
  Aux: Ry(2α)|0⟩ = cos(α)|0⟩ + sin(α)|1⟩,  α = π·T/2.
  Kontrolisano SP|_{aux=0}: aux=0 → state postaje |ψ_sharp⟩ (amp iz CELOG CSV-a).
  Kontrolisani CH|_{aux=1}: aux=1 → state postaje |uniform⟩ = (1/√dim)·Σ|k⟩.
Marginala: p[k] = cos²(α)·|ψ_sharp[k]|² + sin²(α)·(1/2^nq).

T interpretacija (LLM semantika):
  T = 0 → precizno/pouzdano (pure CSV signal).
  T = 0.5 → 50/50 mix.
  T = 1 → uniformno/raznolikost (pure high-temp).

Zašto T nije u grid-pretrazi:
  Za loto-distribuciju (skoro-uniforman freq_csv), cos(bias_uniform, freq_csv_n)
  je sistematski sličan cos(bias_sharp, freq_csv_n), pa bi grid lažno favorizovao
  T = 1 i davao degenerativan NEXT (npr. 1..7 kad je nq = 5 < log2(39)).
  Umesto toga, T je eksplicitan hiperparametar; glavna predikcija ide na T_MAIN
  (default 0.0), a demonstracija kroz niz T_DEMO pokazuje efekat temperature.

Tehnike:
Amplitude encoding (StatePreparation) za sharp režim.
H^⊗nq (preko CH|_{aux=1}) za uniformni režim.
Kontrolisana priprema oba režima iz jednog aux qubit-a.
Egzaktni Statevector (bez uzorkovanja).

Prednosti:
Kvantno (ne klasično softmax-tempiranje) — mešavina kroz aux superpoziciju.
Samo 1 dodatni qubit; vrlo jeftino (nq + 1 qubit-a).
Čisto kvantno: bez klasičnog treninga, bez softmax-a, bez hibrida.

Nedostaci:
Marginala je linearna konveksna kombinacija (aux ortogonalna) — nema interferencije
između režima.
mod-39 readout meša stanja (dim 2^nq ≠ 39); pri nq = 5, 7 pozicija bias-a je 0.
T se ne može automatski birati cos-metrikom nad približno-uniformnim freq_csv.
"""
