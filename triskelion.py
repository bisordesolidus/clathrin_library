r"""
triskelion.py — геометрия одного трискелиона в телесном репере (MODEL.md §3).

Трискелион — жёсткое тело (v1). Телесный репер:
  ê₃ — ось симметрии C₃ хаба, направлена НАРУЖУ от мембраны;
  три ноги под азимутами φ_a = 2πa/3, a = 0,1,2.

Репер ноги a (углы Эйлера ZYZ), трансляция 0:
        h_a = R_z(φ_a) · R_y(ψ) · R_z(χ)  ∈ SO(3).

СОГЛАШЕНИЕ (важно): нога идёт вдоль ЛОКАЛЬНОЙ оси ê₃ репера ноги — дуговая
длина s откладывается по ê₃. Тогда каждый множитель имеет чистый смысл:
  * R_z(χ)  — закрутка вокруг оси ноги, НЕ меняет направление ноги
              ⇒ носитель ХИРАЛЬНОСТИ (ориентация связывающей грани);
  * R_y(ψ)  — наклон ноги от ê₃ на угол пакера ψ;
  * R_z(φ_a) — азимутальная расстановка трёх ног.
Отсюда направление ноги
        û_a = h_a · ê₃ = (sinψ cosφ_a,  sinψ sinφ_a,  cosψ),
а связывающая нормаль
        n̂_a = h_a · ê₁   (несёт χ; n̂_a · φ̂_a = sinχ — инвариант хиральности).

  [Поправка к MODEL.md §3: дуговая ось ноги — ê₃, а не ê₁. При композиции ZYZ
   только так R_z(χ) остаётся чистой закруткой и û_a получается указанной
   формулой; иначе χ подмешивалась бы в направление ноги.]

ψ ≈ 97° > 90°  ⇒  cosψ < 0: ноги отогнуты на ~7° НИЖЕ экватора, к мембране
(сторона −ê₃) — геометрический источник спонтанной кривизны оболочки.

Границы сегментов и контактные сайты — из Morris et al., NSMB 26, 890 (2019).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

E1 = np.array([1.0, 0.0, 0.0])
E2 = np.array([0.0, 1.0, 0.0])
E3 = np.array([0.0, 0.0, 1.0])

# Значения по умолчанию.
PSI_DEG = 110.0          # угол пакера: ИЗМЕРЕНО из PDB 6SCT (проксималь ↔ C₃-ось
                         # наружу, 110.1–110.2° по всем 3 ногам). Лит. «96–98°» —
                         # другое определение угла.
CHI_DEG = -5.0           # закрутка/хиральность: ИЗМЕРЕНО из 6SCT (−5.2° по лёгкой
                         # цепи, −5.4° по лег-лег грани, C₃-консистентно ±0.1°;
                         # хиральность ОТРИЦАТЕЛЬНА). Было +30° (заглушка, неверный знак).
LEG_LENGTH_NM = 50.0     # полная длина ноги, контур (Muthukumar & Nossal 2013)
EDGE_NM = 18.5           # длина ребра = хаб–хаб (Morris 2019, норм. библиотеки)

# Границы сегментов тяжёлой цепи (номера остатков), от хаба к кончику.
SEGMENTS: dict[str, tuple[int, int]] = {
    "txd":      (1576, 1675),   # тримеризация — хаб, не выступающая нога
    "proximal": (1198, 1576),
    "knee":     (1074, 1198),
    "distal":   (838, 1074),
    "ankle":    (330, 838),
    "td":       (1, 330),
}

# Дуговая длина s (нм) вдоль ноги от хаба наружу: (s у большего остатка,
# s у меньшего). Калибровка ИЗ 6SCT: проксималь 1248–1576 = 11.4 нм (прямой
# размах), дистальный 838–1074 = 8.5 нм. Проксималь КОРОЧЕ ребра (18.5) —
# две антипараллельные проксимали перекрываются ~4.3 нм в середине ребра.
# knee/ankle/td — разумные умолчания до полной ноги ~50 нм (контур).
DEFAULT_ARC: dict[str, tuple[float, float]] = {
    "proximal": (0.0, 11.4),
    "knee":     (11.4, 13.4),
    "distal":   (13.4, 21.9),
    "ankle":    (21.9, 40.0),
    "td":       (40.0, 50.0),
}

# Консервативная контактная карта (Morris 2019, «универсальный режим сборки»).
CONTACT_SITES: dict[str, dict] = {
    "D883_888":   dict(segment="distal",   residues=(883, 888),   shell="inner"),
    "D981_984":   dict(segment="distal",   residues=(981, 984),   shell="inner"),
    "D1040_1046": dict(segment="distal",   residues=(1040, 1046), shell="inner"),
    "P1428_1433": dict(segment="proximal", residues=(1428, 1433), shell="outer"),
}
# Какой сайт с каким контактирует (через ребро, между разными трискелионами).
CONTACT_PAIRS: list[tuple[str, str, str]] = [
    ("D883_888",   "D981_984",   "distal-distal"),
    ("D1040_1046", "P1428_1433", "distal-proximal"),
]


def _Rz(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _Ry(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


@dataclass(frozen=True)
class Triskelion:
    r"""Один трискелион в телесном репере. Иммутабелен; хиральность = знак χ."""
    psi: float = np.deg2rad(PSI_DEG)          # угол пакера, рад
    chi: float = np.deg2rad(CHI_DEG)          # закрутка, рад
    arc: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_ARC))

    # ---- реперы и направления ног ---------------------------------------- #
    def leg_rotation(self, a: int) -> np.ndarray:
        r"""R(h_a) = R_z(φ_a) R_y(ψ) R_z(χ) ∈ SO(3), φ_a = 2πa/3."""
        phi_a = 2.0 * np.pi * (a % 3) / 3.0
        return _Rz(phi_a) @ _Ry(self.psi) @ _Rz(self.chi)

    def leg_frame(self, a: int) -> np.ndarray:
        r"""h_a ∈ SE(3) как 4×4 (трансляция 0)."""
        T = np.eye(4)
        T[:3, :3] = self.leg_rotation(a)
        return T

    def leg_axis(self, a: int) -> np.ndarray:
        r"""Направление ноги û_a = R(h_a) ê₃ (не зависит от χ)."""
        return self.leg_rotation(a) @ E3

    def binding_normal(self, a: int) -> np.ndarray:
        r"""Связывающая нормаль n̂_a = R(h_a) ê₁ (несёт χ)."""
        return self.leg_rotation(a) @ E1

    # ---- дуговая длина s(остаток) ---------------------------------------- #
    def arclength(self, residue: float) -> float:
        r"""Остаток → дуговая длина s (нм) вдоль ноги от хаба. Кусочно-линейно.

        Больший остаток ближе к хабу (нога идёт от TxD ~1576 к TD ~1)."""
        for name, (lo, hi) in SEGMENTS.items():
            if name not in self.arc:
                continue
            if lo <= residue <= hi:
                s0, s1 = self.arc[name]          # s0 у остатка hi, s1 у остатка lo
                frac = (hi - residue) / (hi - lo)
                return s0 + frac * (s1 - s0)
        raise ValueError(
            f"остаток {residue} вне выступающей ноги (1..1576); "
            f"остатки 1576..1675 — хаб TxD")

    # ---- точки на ноге --------------------------------------------------- #
    def site_body(self, a: int, s: float,
                  transverse: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        r"""Телесная координата точки на ноге a: ρ = R(h_a)·(t₁ê₁ + t₂ê₂ + s ê₃).

        transverse=(0,0) — на осевой линии ноги: ρ = s·û_a (χ не входит).
        Ненулевой поперечный сдвиг лежит вдоль связывающей нормали n̂_a(χ) и
        именно через него хиральность входит в контактную геометрию (см. M2)."""
        Rot = self.leg_rotation(a)
        local = np.array([transverse[0], transverse[1], float(s)])
        return Rot @ local

    def site_residue(self, a: int, residue: float,
                     transverse: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        r"""То же, но точка задана номером остатка."""
        return self.site_body(a, self.arclength(residue), transverse)

    def contact_site_body(self, a: int, name: str,
                          transverse: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        r"""Телесная координата именованного контактного сайта (Morris 2019)."""
        res = CONTACT_SITES[name]["residues"]
        return self.site_residue(a, 0.5 * (res[0] + res[1]), transverse)

    def mirror(self) -> "Triskelion":
        r"""Зеркальный энантиомер: χ → −χ (при том же ψ)."""
        return Triskelion(psi=self.psi, chi=-self.chi, arc=dict(self.arc))
