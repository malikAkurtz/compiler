"""
harmonic_manim.py  —  Quantum Harmonic Oscillator Educational Video
3Blue1Brown style: geometric intuition, dark background, live codebase physics

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTALL (if needed):
    pip install manim

RUN individual scenes (fast, low-quality preview):
    manim -pql harmonic_manim.py S01_ClassicalSpring
    manim -pql harmonic_manim.py S02_EnergyLevels
    manim -pql harmonic_manim.py S03_Wavefunctions
    manim -pql harmonic_manim.py S04_LadderOperators
    manim -pql harmonic_manim.py S05_NumberAndHamiltonian
    manim -pql harmonic_manim.py S06_TimeEvolution
    manim -pql harmonic_manim.py S07_QubitSubspace

RUN all at high quality then concatenate:
    for s in S01_ClassicalSpring S02_EnergyLevels S03_Wavefunctions \
             S04_LadderOperators S05_NumberAndHamiltonian S06_TimeEvolution \
             S07_QubitSubspace; do
        manim -pqh harmonic_manim.py $s
    done
    # Then use FFmpeg to join the resulting .mp4 files
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.linalg import expm
from scipy.special import eval_hermite
from math import factorial as _fac
from manim import *

# ── Codebase imports ───────────────────────────────────────────
from HarmonicOscillator import HarmonicOscillator
from constants import hbar

# ══════════════════════════════════════════════════════════════
# GLOBAL PHYSICS  (all scenes share one oscillator instance)
# ══════════════════════════════════════════════════════════════
C     = 100e-15                      # [F]   capacitance / mass analog
L     = 10e-9                        # [H]   inductance  / spring analog
OMEGA = np.sqrt(1.0 / (L * C))      # [rad/s]  resonant frequency
X0    = np.sqrt(hbar / (C * OMEGA)) # [m]   zero-point length
T_QUB = 2 * np.pi / OMEGA           # [s]   one qubit period
N_CUT = 8                            # Fock space truncation (matches codebase default)

oscillator = HarmonicOscillator(mass=C, angular_frequency=OMEGA, n_cut=N_CUT)

def ψ(n: int, xi) -> np.ndarray:
    """
    n-th energy eigenstate wavefunction in dimensionless coordinate xi = x/x0.
    Computed analytically (Hermite-Gaussian) — matches the codebase's Fock basis.
    """
    Hn   = eval_hermite(n, np.asarray(xi, dtype=float))
    norm = np.pi ** (-0.25) / np.sqrt(float(2 ** n * _fac(n)))
    return norm * Hn * np.exp(-0.5 * xi ** 2)

# ══════════════════════════════════════════════════════════════
# STYLE
# ══════════════════════════════════════════════════════════════
BG           = "#0d0d1a"
POT_COL      = "#4a9eff"   # potential curve
CLASSIC_COL  = "#ffd700"   # classical energy / ball
QUANTUM_COL  = "#ff79c6"   # quantum highlights
CREATION_COL = "#ff79c6"   # a†
ANNIH_COL    = "#8be9fd"   # a
BLOCH_COL    = "#50fa7b"   # Bloch vector
ZPE_COL      = "#f1fa8c"   # zero-point energy

# One color per Fock level (n = 0 … N_CUT-1)
FOCK_COLS = [BLUE_D, TEAL_D, GREEN_D, YELLOW_D, RED_D, PURPLE_D, MAROON_D, PINK]

def setup(scene: Scene):
    """Apply shared background to any scene."""
    scene.camera.background_color = BG


# ══════════════════════════════════════════════════════════════
# S01 — Classical Spring → Potential Energy Well
# ══════════════════════════════════════════════════════════════
class S01_ClassicalSpring(Scene):
    """
    Opens on a spring-mass system oscillating back and forth.
    Introduces the parabolic potential V = ½mω²x² on the right.
    A golden ball rolls smoothly in the well, showing total mechanical energy.

    [Narration cue] "The harmonic oscillator is everywhere — springs, pendulums,
    LC circuits. Its defining feature: a restoring force proportional to displacement."
    """
    def construct(self):
        setup(self)

        # ── Title card ──────────────────────────────────────────
        title = Text("The Harmonic Oscillator", font_size=54, color=WHITE)
        sub   = Text("from springs  →  quantum bits", font_size=28, color=GRAY_B)
        sub.next_to(title, DOWN, buff=0.35)
        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(sub, shift=UP * 0.2), run_time=0.8)
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(sub))

        # ── Spring-mass diagram (left side) ─────────────────────
        wall = Rectangle(width=0.3, height=3.5, color=GRAY_C, fill_opacity=0.75)
        wall.move_to(LEFT * 5.5)
        floor = Line(LEFT * 6.2, RIGHT * 0.5, color=GRAY_D, stroke_width=1.5)
        floor.shift(DOWN * 1.15)

        def make_spring(x_mass, n_coils=14):
            """Zigzag polyline from wall to mass."""
            x0s, x1s = -5.2, x_mass - 0.33
            pts = [[x0s, 0, 0]]
            span = x1s - x0s
            for k in range(n_coils * 2):
                t  = (k + 1) / (n_coils * 2)
                xp = x0s + t * span
                yp = 0.20 * (1 if k % 2 == 0 else -1)
                pts.append([xp, yp, 0])
            pts.append([x1s, 0, 0])
            return VMobject(stroke_color=GRAY_C, stroke_width=2.5
                            ).set_points_as_corners(pts)

        x_tr   = ValueTracker(1.8)
        spring = always_redraw(lambda: make_spring(x_tr.get_value()))
        block  = always_redraw(lambda: Square(
            side_length=0.65, color=BLUE_D, fill_opacity=0.9
        ).move_to([x_tr.get_value(), 0, 0]))

        eq_line = DashedLine([0, -1.2, 0], [0, 0.7, 0],
                             dash_length=0.12, color=GRAY_B, stroke_opacity=0.45)
        eq_lbl  = MathTex("x=0", font_size=20, color=GRAY_B
                          ).next_to(eq_line, DOWN, buff=0.05)

        self.play(FadeIn(wall), FadeIn(floor), Create(spring), FadeIn(block), run_time=1)
        self.play(FadeIn(eq_line), FadeIn(eq_lbl))

        # Oscillate 2 cycles
        for _ in range(2):
            self.play(x_tr.animate.set_value(3.4), run_time=0.9, rate_func=there_and_back)

        # ── Potential energy curve (right side) ─────────────────
        ax = Axes(
            x_range=[-3.5, 3.5, 1], y_range=[0, 4.2, 1],
            x_length=5.5, y_length=4.2,
            axis_config={"color": GRAY_C, "stroke_opacity": 0.6},
            tips=True
        ).to_edge(RIGHT, buff=0.4).shift(DOWN * 0.3)

        x_lbl = MathTex("x", font_size=22, color=GRAY_C
                        ).next_to(ax.x_axis.get_right(), RIGHT, buff=0.1)
        V_lbl = MathTex("V", font_size=22, color=GRAY_C
                        ).next_to(ax.y_axis.get_top(), UP, buff=0.08)

        V_curve = ax.plot(lambda x: 0.5 * x ** 2,
                          x_range=[-2.85, 2.85],
                          color=POT_COL, stroke_width=3)
        V_eq = MathTex(r"V(x) = \tfrac{1}{2}m\omega^2 x^2",
                       font_size=24, color=POT_COL
                       ).next_to(ax, UP, buff=0.2).shift(LEFT * 0.2)

        self.play(Create(ax), Write(x_lbl), Write(V_lbl), run_time=0.8)
        self.play(Create(V_curve), Write(V_eq), run_time=1)

        # Golden ball rolls in the well in sync with block
        ball = always_redraw(lambda: Dot(
            ax.coords_to_point(x_tr.get_value() - 1.8, 0.5 * (x_tr.get_value() - 1.8) ** 2),
            color=CLASSIC_COL, radius=0.13
        ))
        self.add(ball)

        for _ in range(2):
            self.play(x_tr.animate.set_value(3.4), run_time=1.0, rate_func=there_and_back)

        # Energy label on curve
        e_seg = DashedLine(ax.coords_to_point(-2.5, 3.125),
                           ax.coords_to_point(2.5, 3.125),
                           dash_length=0.12, color=CLASSIC_COL)
        e_lbl = MathTex("E", font_size=24, color=CLASSIC_COL
                        ).next_to(e_seg, RIGHT, buff=0.1)
        self.play(Create(e_seg), Write(e_lbl))
        self.wait(1)


# ══════════════════════════════════════════════════════════════
# S02 — Quantization: Discrete Energy Levels
# ══════════════════════════════════════════════════════════════
class S02_EnergyLevels(Scene):
    """
    Contrast: classically energy slides continuously along a dashed line.
    Quantum mechanics: only discrete rungs are allowed.
    Reveal E_n = (n + ½)ℏω level by level, then spotlight the zero-point energy.

    [Narration cue] "Quantum mechanics changes everything. Energy can't take
    any value — it comes in discrete packets, like rungs on a ladder."
    """
    def construct(self):
        setup(self)

        ax = Axes(
            x_range=[-4.2, 4.2, 1], y_range=[-0.2, 6.2, 1],
            x_length=7.5, y_length=6.5,
            axis_config={"color": GRAY_D, "stroke_opacity": 0.45},
            tips=True
        ).shift(LEFT * 0.5)

        x_lbl = MathTex("x", font_size=24, color=GRAY_D
                        ).next_to(ax.x_axis, RIGHT, buff=0.1)
        E_lbl = MathTex("E", font_size=24, color=GRAY_D
                        ).next_to(ax.y_axis, UP, buff=0.08)

        V_curve = ax.plot(lambda x: 0.5 * x ** 2,
                          x_range=[-3.45, 3.45],
                          color=POT_COL, stroke_width=3)

        self.play(Create(ax), Write(x_lbl), Write(E_lbl), run_time=0.8)
        self.play(Create(V_curve), run_time=1)

        # ── Classical: energy slides freely ──
        e_tr    = ValueTracker(1.2)
        cls_seg = always_redraw(lambda: DashedLine(
            ax.coords_to_point(-np.sqrt(2 * e_tr.get_value()), e_tr.get_value()),
            ax.coords_to_point( np.sqrt(2 * e_tr.get_value()), e_tr.get_value()),
            color=CLASSIC_COL, dash_length=0.12
        ))
        cls_txt = Text("Classical: any energy allowed",
                       font_size=24, color=CLASSIC_COL
                       ).to_edge(RIGHT, buff=0.25).shift(UP * 2.8)

        self.play(Create(cls_seg), Write(cls_txt))
        self.play(e_tr.animate.set_value(4.5), run_time=1.5)
        self.play(e_tr.animate.set_value(0.3), run_time=1.5)
        self.play(FadeOut(cls_seg), FadeOut(cls_txt))

        # ── Quantum: discrete levels ──
        qnt_txt = Text("Quantum: only discrete rungs",
                       font_size=24, color=QUANTUM_COL
                       ).to_edge(RIGHT, buff=0.25).shift(UP * 2.8)
        self.play(Write(qnt_txt))

        level_mobs = VGroup()
        for n in range(5):
            En    = n + 0.5          # dimensionless  E/(ℏω)
            xt    = np.sqrt(2 * En)  # classical turning point
            color = FOCK_COLS[n]

            seg = Line(
                ax.coords_to_point(-xt, En),
                ax.coords_to_point( xt, En),
                color=color, stroke_width=2.8
            )
            lbl = MathTex(
                r"E_{" + str(n) + r"} = \!\left(" + str(n) + r"+ \tfrac{1}{2}\right)\hbar\omega",
                font_size=21, color=color
            ).next_to(seg, RIGHT, buff=0.15)

            level_mobs.add(VGroup(seg, lbl))
            self.play(Create(seg), Write(lbl), run_time=0.5)

        self.wait(0.4)

        # ── Zero-point energy spotlight ──
        zpe_box = SurroundingRectangle(
            level_mobs[0], color=ZPE_COL, buff=0.14, corner_radius=0.1
        )
        zpe_lbl = Text("zero-point energy — the vacuum is not still!",
                       font_size=21, color=ZPE_COL)
        zpe_lbl.next_to(zpe_box, DOWN, buff=0.22)

        self.play(Create(zpe_box), Write(zpe_lbl))
        self.wait(1.8)
        self.play(FadeOut(zpe_box), FadeOut(zpe_lbl))
        self.wait(0.5)


# ══════════════════════════════════════════════════════════════
# S03 — Wavefunctions & Probability Densities
# ══════════════════════════════════════════════════════════════
class S03_Wavefunctions(Scene):
    """
    Plot ψ_n(x) (wavefunction) and |ψ_n(x)|² (probability density) for n = 0…3,
    each overlaid on its energy level.  Highlight: n nodes ↔ n-th level.
    Show the ground-state Gaussian formula.

    [Narration cue] "Each rung has a wavefunction — a probability amplitude
    spread across space.  The ground state is a Gaussian; higher states
    develop oscillating lobes."
    """
    def construct(self):
        setup(self)

        SCALE = 0.52   # vertical display scale for wavefunction amplitude

        ax = Axes(
            x_range=[-4.8, 4.8, 1], y_range=[-0.3, 5.5, 1],
            x_length=9, y_length=6.5,
            axis_config={"color": GRAY_D, "stroke_opacity": 0.3},
            tips=False
        )
        V_curve = ax.plot(lambda x: 0.5 * x ** 2,
                          x_range=[-3.3, 3.3],
                          color=POT_COL, stroke_width=2, stroke_opacity=0.4)
        xi_lbl = MathTex(r"x/x_0", font_size=22, color=GRAY_D
                         ).next_to(ax.x_axis, RIGHT, buff=0.1)
        self.play(Create(ax), Create(V_curve), Write(xi_lbl), run_time=0.8)

        for n in range(4):
            En    = n + 0.5
            xt    = np.sqrt(2 * En)
            color = FOCK_COLS[n]

            # --- baseline at energy level ---
            baseline = Line(
                ax.coords_to_point(-(xt + 0.4), En),
                ax.coords_to_point(  xt + 0.4,  En),
                color=color, stroke_width=1.5, stroke_opacity=0.5
            )

            # --- wavefunction ψ_n(xi) ---
            wf = ax.plot(
                lambda x, _n=n, _E=En: _E + SCALE * ψ(_n, x),
                x_range=[-4.5, 4.5, 0.01],
                color=color, stroke_width=2.5
            )

            # --- |ψ_n|² filled region ---
            xi_pts  = np.linspace(-4.5, 4.5, 300)
            poly_pts = []
            for xi in xi_pts:
                pd = float(ψ(n, xi) ** 2)
                poly_pts.append(ax.coords_to_point(xi, En + SCALE * pd))
            # close back along baseline
            for xi in reversed(xi_pts):
                poly_pts.append(ax.coords_to_point(xi, En))
            fill = Polygon(
                *poly_pts,
                fill_color=color, fill_opacity=0.22,
                stroke_width=0
            )

            n_lbl = MathTex(f"|{n}\\rangle", font_size=30, color=color)
            n_lbl.next_to(ax.coords_to_point(4.6, En), RIGHT, buff=0.05)

            self.play(
                Create(baseline),
                Create(wf),
                FadeIn(fill),
                Write(n_lbl),
                run_time=0.9
            )
            self.wait(0.2)

        # ── Annotate: node count ──
        node_txt = MathTex(r"|n\rangle \text{ has exactly } n \text{ nodes}",
                           font_size=28, color=WHITE).to_edge(DOWN, buff=0.4)
        self.play(Write(node_txt))
        self.wait(1.5)
        self.play(FadeOut(node_txt))

        # ── Ground state Gaussian ──
        gs_eq = MathTex(
            r"\psi_0(x) = \pi^{-1/4}\, e^{-x^2/2}",
            r"\quad\quad x \equiv x/x_0",
            font_size=28, color=FOCK_COLS[0]
        )
        gs_eq.to_edge(DOWN, buff=0.4)
        self.play(Write(gs_eq))
        self.wait(2)


# ══════════════════════════════════════════════════════════════
# S04 — Ladder Operators
# ══════════════════════════════════════════════════════════════
class S04_LadderOperators(Scene):
    """
    Geometric: animated arrows stepping up (a†) and down (a) the energy ladder.
    Algebraic: the actual creation/annihilation matrices from the codebase.
    Key fact: a|0⟩ = 0 — the ground state cannot be lowered.

    [Narration cue] "Meet the ladder operators — the most elegant way to move
    between energy levels.  a† creates a quantum; a destroys one."
    """
    def construct(self):
        setup(self)

        N_SHOW  = 6
        SPACING = 1.05
        BASE_Y  = -2.7
        LVL_X   = LEFT * 3.5

        # ── Draw level ladder ──
        segs, lbls = [], []
        for n in range(N_SHOW):
            y     = BASE_Y + n * SPACING
            color = FOCK_COLS[n]
            seg   = Line(LVL_X + LEFT * 0.9 + UP * y,
                         LVL_X + RIGHT * 0.9 + UP * y,
                         color=color, stroke_width=2.8)
            lbl   = MathTex(f"|{n}\\rangle", font_size=26, color=color)
            lbl.next_to(seg, LEFT, buff=0.18)
            segs.append(seg)
            lbls.append(lbl)

        self.play(*[Create(s) for s in segs], *[Write(l) for l in lbls], run_time=1.2)
        self.wait(0.3)

        # ── a† : creation ──────────────────────────────────────
        adag_hdr = MathTex(r"a^\dagger", font_size=50, color=CREATION_COL
                           ).move_to(RIGHT * 1.5 + UP * 2.5)
        adag_sub = Text("creation  —  raises energy", font_size=24, color=CREATION_COL
                        ).next_to(adag_hdr, DOWN, buff=0.15)
        self.play(Write(adag_hdr), Write(adag_sub))

        up_arrows, up_eqs = [], []
        for n in range(N_SHOW - 1):
            y0 = BASE_Y + n * SPACING
            y1 = BASE_Y + (n + 1) * SPACING
            arr = Arrow(
                LVL_X + UP * y0 + RIGHT * 1.0,
                LVL_X + UP * y1 + RIGHT * 1.0,
                color=CREATION_COL, buff=0.06, stroke_width=2.5
            )
            coeff = MathTex(f"\\sqrt{{{n + 1}}}", font_size=22, color=CREATION_COL
                            ).next_to(arr, RIGHT, buff=0.08)
            eq = MathTex(
                f"a^\\dagger|{n}\\rangle = \\sqrt{{{n + 1}}}\\,|{n + 1}\\rangle",
                font_size=24, color=WHITE
            ).move_to(RIGHT * 3.8 + UP * (BASE_Y + n * SPACING + 0.3))
            up_arrows.append(VGroup(arr, coeff))
            up_eqs.append(eq)
            self.play(GrowArrow(arr), Write(coeff), Write(eq), run_time=0.5)

        self.wait(0.8)
        self.play(
            FadeOut(adag_hdr), FadeOut(adag_sub),
            *[FadeOut(m) for m in up_arrows + up_eqs],
            run_time=0.6
        )

        # ── a : annihilation ────────────────────────────────────
        a_hdr = MathTex(r"a", font_size=50, color=ANNIH_COL
                        ).move_to(RIGHT * 1.5 + UP * 2.5)
        a_sub = Text("annihilation  —  lowers energy", font_size=24, color=ANNIH_COL
                     ).next_to(a_hdr, DOWN, buff=0.15)
        self.play(Write(a_hdr), Write(a_sub))

        dn_arrows = []
        for n in range(1, N_SHOW):
            y0 = BASE_Y + n * SPACING
            y1 = BASE_Y + (n - 1) * SPACING
            arr = Arrow(
                LVL_X + UP * y0 + RIGHT * 1.0,
                LVL_X + UP * y1 + RIGHT * 1.0,
                color=ANNIH_COL, buff=0.06, stroke_width=2.5
            )
            coeff = MathTex(f"\\sqrt{{{n}}}", font_size=22, color=ANNIH_COL
                            ).next_to(arr, RIGHT, buff=0.08)
            dn_arrows.append(VGroup(arr, coeff))
            self.play(GrowArrow(arr), Write(coeff), run_time=0.42)

        # a|0⟩ = 0
        a0_eq  = MathTex(r"a|0\rangle = 0", font_size=38, color=RED_D
                         ).move_to(RIGHT * 3.5 + DOWN * 1.8)
        a0_box = SurroundingRectangle(a0_eq, color=RED_D, buff=0.15, corner_radius=0.1)
        a0_sub = Text("the ground state cannot be lowered",
                      font_size=20, color=RED_D).next_to(a0_box, DOWN, buff=0.15)
        self.play(Write(a0_eq), Create(a0_box), Write(a0_sub))
        self.wait(1)

        # ── Actual matrix from codebase ──────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.7)

        hdr = Text("The annihilation matrix  a  (from codebase, first 5×5):",
                   font_size=26, color=WHITE).to_edge(UP, buff=0.55)
        self.play(Write(hdr))

        # Pull from codebase's HarmonicOscillator
        a_raw = oscillator.annihilation["fock"][:5, :5]
        entries = [
            [f"{a_raw[i, j]:.2g}" if abs(a_raw[i, j]) > 1e-9 else "0"
             for j in range(5)]
            for i in range(5)
        ]
        mat  = Matrix(entries, element_to_mobject_config={"font_size": 28}
                      ).set_color(ANNIH_COL)
        a_lbl = MathTex(r"a =", font_size=38).next_to(mat, LEFT, buff=0.3)
        self.play(Write(a_lbl), Create(mat), run_time=1.5)

        # Pattern annotation
        pattern = MathTex(r"a_{m,\,n} = \sqrt{n}\;\delta_{m,\;n-1}",
                          font_size=34, color=YELLOW_D)
        pattern.to_edge(DOWN, buff=0.5)
        self.play(Write(pattern))
        self.wait(0.5)

        # Similarly show a† = aT
        adag_note = MathTex(r"a^\dagger = a^T \quad \text{(real transpose)}",
                            font_size=28, color=CREATION_COL)
        adag_note.next_to(pattern, UP, buff=0.3)
        self.play(Write(adag_note))
        self.wait(2)


# ══════════════════════════════════════════════════════════════
# S05 — Number Operator & Hamiltonian
# ══════════════════════════════════════════════════════════════
class S05_NumberAndHamiltonian(Scene):
    """
    Build N̂ = a†a from pieces, show its diagonal matrix (counts quanta).
    Then construct H = ℏω(N̂ + ½) and display the Hamiltonian matrix
    pulled directly from the codebase.

    [Narration cue] "Combine the two ladders.  a†a counts how many quanta
    are present — the number operator.  The energy is just ℏω times that count,
    plus the unavoidable half-quantum."
    """
    def construct(self):
        setup(self)

        # ── Build up N̂ = a†a ──
        eq_adag = MathTex(r"a^\dagger", font_size=52, color=CREATION_COL)
        eq_a    = MathTex(r"a",         font_size=52, color=ANNIH_COL)
        eq_N    = MathTex(r"\hat{N} = a^\dagger a", font_size=52, color=WHITE)

        eq_adag.move_to(LEFT * 1.5 + UP * 2)
        eq_a.move_to(RIGHT * 1.5 + UP * 2)

        self.play(Write(eq_adag), Write(eq_a))
        self.wait(0.4)
        self.play(ReplacementTransform(VGroup(eq_adag, eq_a), eq_N))
        self.wait(0.5)
        self.play(eq_N.animate.to_edge(UP, buff=0.5).scale(0.85))

        # ── N̂ matrix (diagonal — counts quanta) ──
        N_raw = oscillator.N["fock"][:5, :5].real
        entries_N = [
            [str(int(round(N_raw[i, j]))) for j in range(5)]
            for i in range(5)
        ]
        mat_N   = Matrix(entries_N, element_to_mobject_config={"font_size": 30}
                         ).set_color(GREEN_D).shift(LEFT * 2.8 + UP * 0.2)
        lbl_N   = MathTex(r"\hat{N} =", font_size=36).next_to(mat_N, LEFT)
        diag_N  = Text("diagonal entries: 0, 1, 2, 3, 4 …",
                       font_size=22, color=GREEN_D
                       ).next_to(mat_N, DOWN, buff=0.25)

        self.play(Write(lbl_N), Create(mat_N), run_time=1.5)
        self.play(Write(diag_N))

        eigen = MathTex(r"\hat{N}|n\rangle = n\,|n\rangle",
                        font_size=34, color=GREEN_D)
        eigen.move_to(RIGHT * 3.2 + UP * 1.0)
        self.play(Write(eigen))
        self.wait(0.5)

        # ── Hamiltonian ──
        H_eq = MathTex(
            r"H = \hbar\omega\!\left(\hat{N} + \tfrac{1}{2}\right)",
            font_size=44, color=YELLOW_D
        ).move_to(RIGHT * 3.2 + DOWN * 0.3)
        self.play(Write(H_eq))
        self.wait(0.4)

        # H matrix from codebase (normalised by ℏω)
        H_raw = oscillator.H0["fock"][:5, :5].real / (hbar * OMEGA)
        entries_H = [
            [f"{H_raw[i,j]:.1f}" if abs(H_raw[i,j]) > 1e-9 else "0"
             for j in range(5)]
            for i in range(5)
        ]
        mat_H = Matrix(entries_H, element_to_mobject_config={"font_size": 26}
                       ).set_color(YELLOW_D).shift(LEFT * 2.8 + DOWN * 2.0)
        lbl_H = MathTex(r"\tfrac{H}{\hbar\omega} =", font_size=30
                        ).next_to(mat_H, LEFT)

        self.play(Write(lbl_H), Create(mat_H), run_time=1.5)

        diag_H = Text("diagonal: n + ½", font_size=22, color=YELLOW_D
                      ).next_to(mat_H, DOWN, buff=0.2)
        self.play(Write(diag_H))
        self.wait(2)


# ══════════════════════════════════════════════════════════════
# S06 — Time Evolution
# ══════════════════════════════════════════════════════════════
class S06_TimeEvolution(Scene):
    """
    Two demos using codebase's H0 matrix:
    1. Superposition |0⟩ + |1⟩: populations oscillate between the two levels.
    2. Coherent state |α⟩: Poisson-distributed populations shift classically.

    [Narration cue] "Time evolution is governed by the Schrödinger equation.
    Feed in H, exponentiate, and watch energy slosh between levels."
    """
    def construct(self):
        setup(self)

        H0 = oscillator.H0["fock"].real

        def evolve(coefs: np.ndarray, t: float) -> np.ndarray:
            return expm(-1j * H0 * t / hbar) @ coefs

        # ── Layout ──
        ax = Axes(
            x_range=[-0.5, N_CUT - 0.5, 1],
            y_range=[0, 1.05, 0.25],
            x_length=8.5, y_length=4.0,
            axis_config={"color": GRAY_D},
            y_axis_config={"include_tip": False},
        ).shift(DOWN * 0.9)

        x_lbl = Text("Fock state  |n⟩", font_size=22, color=GRAY_D
                     ).next_to(ax, DOWN, buff=0.3)
        y_lbl = MathTex(r"|\langle n|\psi\rangle|^2", font_size=22, color=GRAY_D
                        ).next_to(ax.y_axis, LEFT, buff=0.15)

        # Fock-state tick labels
        tick_lbls = VGroup(*[
            MathTex(f"|{n}\\rangle", font_size=20,
                    color=FOCK_COLS[n % len(FOCK_COLS)])
            .next_to(ax.coords_to_point(n, 0), DOWN, buff=0.25)
            for n in range(N_CUT)
        ])

        # ── Helper: build bar chart from probability vector ──
        BAR_W = 0.7
        BAR_H = 3.8    # pixels for probability = 1

        def make_bars(probs: np.ndarray) -> VGroup:
            bars = VGroup()
            for n, p in enumerate(probs):
                h = max(float(p), 1e-4) * BAR_H
                bar = Rectangle(
                    width=BAR_W, height=h,
                    fill_color=FOCK_COLS[n % len(FOCK_COLS)],
                    fill_opacity=0.85, stroke_width=1, stroke_color=WHITE
                ).next_to(ax.coords_to_point(n, 0), UP, buff=0)
                bars.add(bar)
            return bars

        # ══ Demo 1: |0⟩ + |1⟩ superposition ══════════════════
        title = MathTex(
            r"|\psi(t)\rangle = e^{-iHt/\hbar}\,|\psi(0)\rangle",
            font_size=34, color=WHITE
        ).to_edge(UP, buff=0.5)

        c_super      = np.zeros(N_CUT, dtype=complex)
        c_super[0]   = c_super[1] = 1 / np.sqrt(2)

        t_tr = ValueTracker(0.0)

        time_disp = always_redraw(lambda: MathTex(
            r"t = " + f"{t_tr.get_value() / T_QUB:.2f}" + r"\,T",
            font_size=28, color=YELLOW_D
        ).to_edge(RIGHT, buff=0.6).shift(UP * 2.5))

        bars = always_redraw(
            lambda: make_bars(np.abs(evolve(c_super, t_tr.get_value())) ** 2)
        )

        # initial state annotation
        init_txt = MathTex(
            r"|\psi(0)\rangle = \tfrac{1}{\sqrt{2}}\bigl(|0\rangle + |1\rangle\bigr)",
            font_size=26, color=WHITE
        ).to_edge(RIGHT, buff=0.3).shift(UP * 1.2)

        self.play(Write(title), Create(ax), Write(x_lbl), Write(y_lbl), run_time=0.9)
        self.add(bars, time_disp, tick_lbls)
        self.play(FadeIn(init_txt))
        self.wait(0.8)

        # Animate 2 full periods  (populations oscillate between |0⟩ and |1⟩)
        self.play(t_tr.animate.set_value(2 * T_QUB), run_time=7, rate_func=linear)
        self.wait(0.5)

        # ══ Demo 2: coherent state ════════════════════════════
        coher_title = Text("Coherent state  |α⟩  — most classical-like",
                           font_size=28, color=CLASSIC_COL).to_edge(UP, buff=0.5)
        self.play(
            ReplacementTransform(title, coher_title),
            FadeOut(init_txt)
        )

        alpha_sq = 2.5    # mean photon number |α|²
        c_coh = np.array([
            np.exp(-alpha_sq / 2) * (alpha_sq ** (n / 2)) / np.sqrt(float(_fac(n)))
            for n in range(N_CUT)
        ], dtype=complex)
        c_coh /= np.linalg.norm(c_coh)

        poisson_txt = MathTex(
            r"P_n = e^{-|\alpha|^2} \frac{|\alpha|^{2n}}{n!}",
            font_size=28, color=CLASSIC_COL
        ).to_edge(RIGHT, buff=0.3).shift(UP * 1.2)
        self.play(Write(poisson_txt))

        t_tr.set_value(0)
        coh_bars = always_redraw(
            lambda: make_bars(np.abs(evolve(c_coh, t_tr.get_value())) ** 2)
        )

        self.play(ReplacementTransform(bars, coh_bars), run_time=1)
        self.play(t_tr.animate.set_value(2 * T_QUB), run_time=7, rate_func=linear)
        self.wait(1)


# ══════════════════════════════════════════════════════════════
# S07 — The Qubit: Truncating to |0⟩ and |1⟩
# ══════════════════════════════════════════════════════════════
class S07_QubitSubspace(Scene):
    """
    Dim out n ≥ 2 levels to reveal the two-level qubit.
    Draw a 2D Bloch sphere projection; animate the state vector tracing
    circles for free precession, then snap to specific gates.
    Show how the harmonic oscillator period T sets the gate clock.

    [Narration cue] "Keep only the bottom two rungs.  Suddenly the oscillator
    looks like a qubit — a |0⟩ and a |1⟩.  A spinning arrow on the Bloch
    sphere tells us everything about its state."
    """
    def construct(self):
        setup(self)

        # ── Energy ladder (left panel) ──
        N_SHOW  = 5
        SPACING = 1.1
        BASE_Y  = -2.4
        LVL_X   = LEFT * 3.6

        lvl_mobs = []
        for n in range(N_SHOW):
            y     = BASE_Y + n * SPACING
            color = FOCK_COLS[n]
            seg   = Line(LVL_X + LEFT * 0.9 + UP * y,
                         LVL_X + RIGHT * 0.9 + UP * y,
                         color=color, stroke_width=2.8)
            lbl   = MathTex(f"|{n}\\rangle", font_size=26, color=color)
            lbl.next_to(seg, LEFT, buff=0.18)
            lvl_mobs.append(VGroup(seg, lbl))

        self.play(*[Create(m) for m in lvl_mobs], run_time=1.2)
        self.wait(0.3)

        # Highlight qubit subspace
        qubit_box = SurroundingRectangle(
            VGroup(lvl_mobs[0], lvl_mobs[1]),
            color=WHITE, buff=0.18, corner_radius=0.12
        )
        qubit_lbl = Text("computational\nsubspace", font_size=22, color=WHITE,
                         line_spacing=0.8).next_to(qubit_box, RIGHT, buff=0.25)
        self.play(Create(qubit_box), Write(qubit_lbl))

        # Dim higher levels
        self.play(
            *[m.animate.set_opacity(0.18) for m in lvl_mobs[2:]],
            run_time=0.8
        )
        self.wait(0.4)

        # ── 2-D Bloch sphere projection (right panel) ──────────
        R        = 1.9
        CENTER   = RIGHT * 3.0 + UP * 0.1

        sphere_circle = Circle(radius=R, color=GRAY_B, stroke_opacity=0.4)
        sphere_circle.move_to(CENTER)

        # Axes lines
        z_axis = Line(CENTER + DOWN * (R + 0.2), CENTER + UP * (R + 0.2),
                      color=GRAY_C, stroke_width=1.2)
        x_axis = Line(CENTER + LEFT * (R + 0.2), CENTER + RIGHT * (R + 0.2),
                      color=GRAY_C, stroke_width=1.2)
        # Equator (shows XY plane projection)
        equator = Ellipse(width=2 * R, height=0.55 * R,
                          color=GRAY_D, stroke_opacity=0.35)
        equator.move_to(CENTER)

        z0_lbl = MathTex(r"|0\rangle", font_size=26, color=BLUE_D
                         ).next_to(CENTER + UP * R, UP, buff=0.08)
        z1_lbl = MathTex(r"|1\rangle", font_size=26, color=RED_D
                         ).next_to(CENTER + DOWN * R, DOWN, buff=0.08)
        x_lbl  = MathTex(r"X", font_size=22, color=GRAY_C
                         ).next_to(CENTER + RIGHT * R, RIGHT, buff=0.08)

        self.play(
            Create(sphere_circle), Create(z_axis), Create(x_axis), Create(equator),
            Write(z0_lbl), Write(z1_lbl), Write(x_lbl),
            run_time=1
        )

        # Bloch vector
        th_tr  = ValueTracker(0.0)   # polar angle  (0 = north pole = |0⟩)
        phi_tr = ValueTracker(0.0)   # azimuthal angle

        def bloch_tip():
            th  = th_tr.get_value()
            phi = phi_tr.get_value()
            bx  = R * np.sin(th) * np.cos(phi)
            by  = R * np.sin(th) * np.sin(phi) * 0.29   # depth foreshortening
            bz  = R * np.cos(th)
            return CENTER + np.array([bx, bz, 0])        # (x→horizontal, z→vertical)

        bloch_vec = always_redraw(lambda: Arrow(
            CENTER, bloch_tip(),
            color=BLOCH_COL, stroke_width=4, buff=0,
            max_tip_length_to_length_ratio=0.12
        ))
        bloch_dot = always_redraw(lambda: Dot(bloch_tip(), color=BLOCH_COL, radius=0.1))

        self.play(FadeIn(bloch_vec), FadeIn(bloch_dot))
        self.wait(0.3)

        # |0⟩ → equator  (prepare superposition)
        self.play(th_tr.animate.set_value(PI / 2), run_time=1.5)

        # Precess around Z axis (free evolution)
        prec_txt = Text("free precession  →  phase", font_size=22, color=QUANTUM_COL
                        ).to_edge(DOWN, buff=1.1)
        self.play(Write(prec_txt))
        self.play(phi_tr.animate.set_value(2 * PI), run_time=3.0, rate_func=linear)
        self.play(FadeOut(prec_txt))

        # General state equation
        state_eq = MathTex(
            r"|\psi\rangle = \alpha|0\rangle + \beta|1\rangle",
            r"\quad |\alpha|^2 + |\beta|^2 = 1",
            font_size=28, color=WHITE
        ).to_edge(DOWN, buff=0.55)
        self.play(Write(state_eq))

        # Demonstrate a few gate-like rotations
        self.play(phi_tr.animate.set_value(PI / 2), run_time=0.8)   # to Y axis
        self.play(th_tr.animate.set_value(PI),       run_time=1.2)   # to |1⟩
        gate_lbl = MathTex(r"X\text{ gate}: |0\rangle \to |1\rangle",
                           font_size=24, color=BLOCH_COL).to_edge(RIGHT, buff=0.3)
        self.play(Write(gate_lbl))
        self.wait(0.5)
        self.play(th_tr.animate.set_value(0),        run_time=1.2)   # back to |0⟩
        self.play(FadeOut(gate_lbl))

        # Period label
        T_lbl = MathTex(
            r"T = \frac{2\pi}{\omega}",
            r"\approx " + f"{T_QUB * 1e9:.2f}" + r"\;\text{ns}",
            font_size=26, color=YELLOW_D
        ).to_edge(RIGHT, buff=0.3).shift(UP * 1.5)
        self.play(Write(T_lbl))
        self.wait(2)


# ══════════════════════════════════════════════════════════════
# CONCATENATION NOTE
# ══════════════════════════════════════════════════════════════
"""
After rendering all 7 scenes at high quality, join them with FFmpeg:

    ffmpeg -f concat -safe 0 \
        -i <(printf "file '%s'\n" \
             media/videos/harmonic_manim/1080p60/S01_ClassicalSpring.mp4 \
             media/videos/harmonic_manim/1080p60/S02_EnergyLevels.mp4 \
             media/videos/harmonic_manim/1080p60/S03_Wavefunctions.mp4 \
             media/videos/harmonic_manim/1080p60/S04_LadderOperators.mp4 \
             media/videos/harmonic_manim/1080p60/S05_NumberAndHamiltonian.mp4 \
             media/videos/harmonic_manim/1080p60/S06_TimeEvolution.mp4 \
             media/videos/harmonic_manim/1080p60/S07_QubitSubspace.mp4) \
        -c copy harmonic_oscillator_full.mp4

Or add transitions:
    manim -pqh harmonic_manim.py S01_ClassicalSpring
    # etc., then use DaVinci Resolve / Premiere for polish
"""
