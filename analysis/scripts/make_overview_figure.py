#!/usr/bin/env python3
"""Overview figure (Fig. 1) for the TRACE ranking-failure story.

(a) One measured mixture (door wood creaks + footsteps, Qwen3-Omni-30B,
condition esc50_0001_q2_snrp0_partial_d1r1): the shared question is scored on
the mixture, the target-removed view (label: absent), and the decisive
rival-removed view (label: present), giving margins +3.63/+2.87/+0.88.  Rather
than three detached score badges -- which left the contradiction to the caption
-- the scores are tabulated as the predicted label in each column under the two
candidate thresholds on the shared margin.  At tau = 0 (as deployed) all three
views are called present, so the target-absent view is wrong; raising tau to
3.0 repairs that view but then the +0.88 view, whose label is present, falls to
absent.  Neither row is free of error, which is the paper's point: no single
threshold reconciles the three views.  Hence the target-only check passes
(Delta_t = +0.75) while the rival-removal effect is larger (Delta_1 = +2.75).
This is the most accurate model in the panel (84.3%), so the conflict is not a
weakness of a single adapted checkpoint.

(b) All correct answers of the eight-model panel plotted against the SAME two
quantities in every region -- the target-removal effect Delta_t (x) and the
strongest rival effect max_j Delta_j (y).  Because the y quantity no longer
changes definition from region to region, the three groups the paper cares
about separate geometrically, with no region labels needed to explain them:

    left of Delta_t = 0                  -> ungrounded (target-removal fails)
    right of 0 and on/above y = x        -> competitor-confused (a rival wins)
    right of 0 and below y = x           -> ordering-consistent

The previous version plotted Delta^+ where it reached Delta_t and otherwise
the signed Delta_j of largest magnitude, so a point's y value meant two
different things depending on where it sat; readers had to hold four shaded
regions and their labels in mind at once.  Keeping y = max_j Delta_j fixed
makes the dashed diagonal the only boundary that needs explaining.

Stems are recovered from the frozen rendered views
(stem = full - view_without_stem).

Layout note: the axes now fill the canvas top-to-bottom (the previous
top=.88/bottom=.14 left ~50 pt of blank inside the canvas, which surfaced as
a visible hole between panel (a) and the caption).  Panel (a) keeps the two
stems as shared row labels on the far left rather than repeating a label
beside every column.
"""
import json, wave
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; G = ROOT / 'generated'
# Case audio lives in data/, not in a scratch tmp/: the three frozen views of
# esc50_0001_q2_snrp0_partial_d1r1 are frozen inputs of Fig. 1, so they ship
# with the analysis instead of being fetched into a throwaway directory.
AUD = ROOT / 'data' / 'case_audio_creak'  # esc50_0001_q2_snrp0_partial_d1r1 frozen views
                                    # (hashes match audit_raw_v3 baseline manifest)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'axes.linewidth': .65, 'savefig.dpi': 300, 'mathtext.fontset': 'dejavusans'})
teal, orange, gray, purple, red = '#007C83', '#C2571A', '#45505B', '#6A4C93', '#B22222'
dark = '#2E3A46'

inp = json.loads((G / 'counterfactual_inputs.json').read_text())
us = inp['panels']['baseline']
for u in us:
    dt = u['mf'] - u['mt']; dj = [u['mf'] - m for m in u['mj']]
    u['dt'] = dt; u['maxdj'] = max(dj); u['maxabs'] = max(abs(x) for x in dj)
    H = dt - max(0, u['maxdj']); Gv = dt - u['maxabs']
    if u['mf'] <= 0: u['state'] = None
    elif dt <= 0: u['state'] = 'ungrounded'
    elif H <= 0: u['state'] = 'confused'
    elif Gv <= 0: u['state'] = 'context'
    else: u['state'] = 'grounded'
    u['H'] = H
corr = [u for u in us if u['state']]
fail = [u for u in corr if u['H'] <= 0]
cons = [u for u in corr if u['H'] > 0]
ung = [u for u in fail if u['dt'] <= 0]
con = [u for u in fail if u['dt'] > 0]
case = next(u for u in us if u['model_id'] == 'qwen3_omni_30b'
            and u['condition_id'] == 'esc50_0001_q2_snrp0_partial_d1r1')
mf, mt, mj = case['mf'], case['mt'], case['mj'][0]
dt = mf - mt; dj = mf - mj
N = len(corr)


def xy(pts):
    return np.array([u['dt'] for u in pts]), np.array([u['maxdj'] for u in pts])


# ---- stems recovered from the frozen views: stem = full - view_without_stem
def rd(p):
    w = wave.open(str(p)); n = w.getnframes()
    a = np.frombuffer(w.readframes(n), dtype=np.int16).astype(float) / 32768
    return a.reshape(-1, w.getnchannels()).mean(1)
def envelope(sig, n=120):
    seg = np.array_split(sig, n)
    return np.array([np.abs(s).max() for s in seg])
full = rd(AUD / 'full.wav')
env_t = envelope(full - rd(AUD / 'remove_target.wav'))   # door wood creaks stem
env_r = envelope(full - rd(AUD / 'remove_off_1.wav'))    # footsteps stem
env_t = env_t / env_t.max(); env_r = env_r / env_r.max()
STEMS = ['door creaks', 'footsteps']
STEM_COL = {'door creaks': teal, 'footsteps': '#8C6A3D'}

fig = plt.figure(figsize=(7.05, 2.16))
# width_ratios is set to the value that centres the gutter on the canvas and
# equalises the two panels' ink, rather than to an arbitrary 1.30:1: the two
# panels' actual ink is 338px (a) vs 313px (b), so a 1.30 budget left (a) with
# slack on its right, pushed (b) outward and put the gutter 13.0px right of the
# canvas centre.  At 1.20 the gutter centre is +0.5px from the canvas centre and
# the inks are 327 vs 326px -- both defects fixed by one parameter, with no
# artist moved and no height change.  Measured, not eyeballed: the panel inks
# are found from the rendered canvas, and the gutter is the longest blank run.
gs = fig.add_gridspec(1, 2, width_ratios=[1.20, 1.0], left=.004, right=.985,
                      top=.905, bottom=.012, wspace=.19)  # top leaves a row for the (b) title, which
#                                            the checker otherwise flags at -1.9pt

# ---- (a) three views of one measured case
ax = fig.add_subplot(gs[0]); ax.set_axis_off(); ax.set_xlim(-17, 100); ax.set_ylim(0, 100)
views = [('mixture $x$', ['door creaks', 'footsteps'], mf, 'present', True),
         (r'$x\setminus$door creaks', ['footsteps'], mt, 'absent', False),
         (r'$x\setminus$footsteps', ['door creaks'], mj, 'present', False)]
ENVS = {'door creaks': env_t, 'footsteps': env_r}
col_w, gap, row_h, top, x_start = 25.5, 3.0, 18.0, 76.0, 14.0
box_bot = top - 2 * row_h - 1.4
xc = [x_start + i * (col_w + gap) + col_w / 2 for i in range(3)]
x_content_right = x_start + 3 * col_w + 2 * gap
# shared row labels, drawn once at the far left (see module docstring); the
# table's own row headers join them, because the left-edge crop below measures
# this list to decide where the panel's ink actually starts
row_lab = []
for r, s in enumerate(STEMS):
    row_lab.append(ax.text(x_start - 1.5, top - (r + .5) * row_h, s, ha='right',
                           va='center', fontsize=6.6, color=STEM_COL[s]))
for i, (name, present, m, label, is_mix) in enumerate(views):
    x0 = x_start + i * (col_w + gap)
    ax.add_patch(FancyBboxPatch((x0, box_bot), col_w, 2 * row_h + 2.8,
                                boxstyle='round,pad=0.3,rounding_size=1.2',
                                fc='#EAF4F4' if is_mix else '#F7F8F9', ec=teal if is_mix else '#B8BEC5', lw=.7))
    ax.text(x0 + col_w / 2, top + 6.0, name, ha='center', va='center', fontsize=6.8,
            fontweight='bold' if is_mix else 'normal', color=teal if is_mix else 'black')
    for r, s in enumerate(STEMS):
        env = ENVS[s] * row_h * .40
        t = np.linspace(x0 + 1.4, x0 + col_w - 1.4, len(env))
        yc = top - (r + .5) * row_h
        if s in present:
            ax.fill_between(t, yc - env, yc + env, color=STEM_COL[s], lw=0, alpha=.95)
        else:
            ax.fill_between(t, yc - env, yc + env, color='#D5D8DC', lw=0, alpha=.9)
            ax.plot([x0 + 1.4, x0 + col_w - 1.4], [yc, yc], color='#9AA0A6', lw=.6)
            ax.text(x0 + col_w / 2, yc, 'removed', ha='center', va='center', fontsize=5.6,
                    color='#5F646B', style='italic', bbox=dict(fc='white', ec='none', pad=.5, alpha=.85))
    # the score badge sits clear of the view box: box bottom is
    # top - 2*row_h - 1.4, so the badge centre is dropped well below it.
    # It is deliberately neutral in colour: the only colour that carries a
    # judgement in this panel is the pass/fail mark in the table below.
    yb = box_bot - 6.0
    ax.add_patch(FancyBboxPatch((x0 + 1.6, yb - 2.9), col_w - 3.2, 5.8,
                                boxstyle='round,pad=0.2,rounding_size=0.9',
                                fc='#E9ECEF', ec='none'))
    ax.text(x0 + col_w / 2, yb, f'$M={m:+.2f}$', ha='center', va='center',
            fontsize=7.4, color=dark, fontweight='bold')
# The table below is the panel's actual argument.  The three margins are fixed;
# what changes between rows is the threshold they are read against.  Each row
# misclassifies exactly one view -- tau=0 the target-removed view (scored above
# zero although its label is absent), tau=3.0 the rival-removed view (pushed to
# absent although its label is present) -- so the wrong cells are the point: no
# single threshold reconciles ground truth with these scores.
y_lab, y_t0, y_t3 = box_bot - 13.4, box_bot - 19.6, box_bot - 25.8
ax.plot([x_start - 1.5, x_content_right], [y_lab + 3.5] * 2, color='#C9CED4', lw=.5)
for y, hdr, tau in ((y_lab, 'label', None), (y_t0, r'$\tau=0$', 0.0),
                    (y_t3, r'$\tau=3.0$', 3.0)):
    row_lab.append(ax.text(x_start - 1.5, y, hdr, ha='right', va='center',
                           fontsize=6.1, color=gray))
    for i, (name, present, m, label, is_mix) in enumerate(views):
        if tau is None:  # ground truth
            ax.text(xc[i], y, label, ha='center', va='center', fontsize=6.3, color=dark)
            continue
        pred = 'present' if m > tau else 'absent'
        ok = pred == label
        ax.text(xc[i], y, pred + ('  \u2713' if ok else '  \u2717'), ha='center', va='center',
                fontsize=6.3, color=gray if ok else red,
                fontweight='normal' if ok else 'bold')
yp = box_bot - 33.0
ax.add_patch(FancyBboxPatch((x_start - 0.5, yp - 3.4), x_content_right - x_start + 1.0, 6.8,
                            boxstyle='round,pad=0.3,rounding_size=1.0', fc='#F1F3F5', ec='none'))
ax.text((x_start + x_content_right) / 2, yp,
        f'$\\Delta_t={dt:+.2f}$ but $\\Delta_1={dj:+.2f}$ $\\Rightarrow$ no single $\\tau$ fits all three',
        ha='center', va='center', fontsize=6.2, color=dark)
# The two panel headings must match: (b) is drawn with ax.set_title(loc='left'),
# which carries fontweight='bold', so (a) -- drawn here by hand -- is given the
# same weight and size.  Without this the two halves of one figure read at
# different emphasis for no reason, since the difference came only from the two
# titles being emitted by different code paths.  t_form below stays regular:
# mathtext ignores fontweight entirely (measured: 0.0px width change), which
# leaves the heading bold and its supporting definition plain.
t_title = ax.text(x_start, 97.5, '(a) "Is there a door wood creaks sound in this audio?"',
                  ha='center', va='center', fontsize=6.9, color='black',
                  fontweight='bold')
t_form = ax.text(x_start, 90.5, r'$M=\log p(\mathrm{present})-\log p(\mathrm{absent})$',
                 ha='center', va='center', fontsize=6.8, color='black')

# Panel (a) is text-anchored, so its widest slack is on the left: the axis used
# to start at -17 while the leftmost ink (the row labels) sits near x=4, wasting
# ~17% of the panel on dead space.  Measure the real label extent and start the
# axis just beside it, then re-centre the two header lines on the content.
fig.canvas.draw()
_inv = ax.transData.inverted()
_x_left = min(_inv.transform((t.get_window_extent(fig.canvas.get_renderer()).x0, 0))[0]
              for t in row_lab)
ax.set_xlim(_x_left - 1.0, 100)
_xc = (_x_left - 1.0 + x_content_right) / 2
t_title.set_position((_xc, 97.5)); t_form.set_position((_xc, 90.5))

# ---- (b) all correct answers: one consistent y quantity, so the three groups
#      separate geometrically instead of needing region labels
ax = fig.add_subplot(gs[1])
# Panel (b) needs its own bottom margin: the gridspec's bottom=.012 leaves only
# 1.9pt below the axes, but the x tick labels need 10.2pt and the x-axis label
# 19.2pt, so matplotlib drew both OFF the canvas and they silently disappeared
# -- the published figure had a fully labelled y-axis and a bare x-axis.
# (check_figure_layout.py cannot see this: an artist drawn outside the canvas is
# never emitted to the PDF, so there is no text for the checker to measure.)
# The 155.5pt canvas height is a hard 5-page constraint, so the room is taken
# from panel (b)'s own height (138.9pt -> 117.4pt).  Only (b) moves; (a) is
# untouched.  The guard at the end of this file asserts both axes stay inside.
PB_BOTTOM = 0.150
_p0 = ax.get_position()
ax.set_position([_p0.x0, PB_BOTTOM, _p0.width, _p0.y1 - PB_BOTTOM])
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
lim = 16; LT = 1.0
ax.set_xscale('symlog', linthresh=LT, linscale=.6); ax.set_yscale('symlog', linthresh=LT, linscale=.6)
XL, YL = (-6, 16), (-10, 10)
# the two failure regions, shaded as the two sides of the audit
ax.axvspan(XL[0], 0, fc='#FBE3E3', ec='none', zorder=0)
ax.add_patch(Polygon([[0, 0], [XL[1], XL[1]], [0, YL[1]]], closed=True,
                     fc='#FBEEDF', ec='none', zorder=0))
ax.axhline(0, color=gray, lw=.55, zorder=1); ax.axvline(0, color=gray, lw=.55, zorder=1)
ax.plot([0, XL[1]], [0, XL[1]], color=red, lw=.85, ls='--', zorder=1)
cx, cy = xy(cons); ux, uy = xy(ung); kx, ky = xy(con)
# Margins are stored in 1/16 steps, so max_j Delta_j takes only 199 distinct
# values over 18,911 points and 11 of those rows carry 46.5% of them.  Drawn as
# small opaque dots the shared rows read as spurious horizontal stripes rather
# than as a density cloud.  Larger, fainter markers let neighbouring grid rows
# blend into one cloud.  This only changes mark size/alpha: no point moves, so
# the three regions and the dashed boundary line are exactly as before.
# Deliberately NOT jitter: 8.9% of the competitor-confused points sit exactly on
# y=x and 15.9% within one grid step of it, so any jitter strong enough to break
# the stripes would push real failures across the line that defines the regions.
ax.scatter(cx, cy, s=4.5, c='#9AA3AD', alpha=.07, linewidths=0, rasterized=True, zorder=2,
           label=f'ordering-consistent  {100*len(cons)/N:.1f}%')
ax.scatter(ux, uy, s=4.5, c=red, alpha=.75, linewidths=0, rasterized=True, zorder=3,
           marker='o', label=f'ungrounded  {100*len(ung)/N:.1f}%')
ax.scatter(kx, ky, s=5.5, c=orange, alpha=.80, linewidths=0, rasterized=True, zorder=3,
           marker='^', label=f'competitor-confused  {100*len(con)/N:.1f}%')
# The legend is kept small and in the conventional lower-right corner.  A grid
# search over placements showed matplotlib's other corners are worse on the
# quantity that matters: 'upper left' hides 31 ungrounded and 16
# competitor-confused points -- the individual failure cases this panel exists to
# display -- whereas this box hides none of them.  Shrinking the box (tighter
# labelspacing/borderpad) also drops the ordering-consistent points it covers
# from ~2,611 to ~1,016, which is a density cloud rather than evidence.
lg = ax.legend(fontsize=6.0, frameon=True, loc='lower right', handlelength=.8,
               labelspacing=.18, borderpad=.25, markerscale=1.2)
lg.get_frame().set_facecolor('white'); lg.get_frame().set_edgecolor('none'); lg.get_frame().set_alpha(.75)
ax.set_xlim(*XL); ax.set_ylim(*YL)
ax.set_xticks([-3, -1, 0, 1, 3, 10]); ax.set_yticks([-3, -1, 0, 1, 3, 10])
ax.set_xticklabels(['-3', '-1', '0', '1', '3', '10']); ax.set_yticklabels(['-3', '-1', '0', '1', '3', '10'])
ax.minorticks_off()
ax.set_xlabel('target-removal effect $\\Delta_t$', fontsize=7.2, labelpad=1)
ax.set_ylabel('strongest rival effect $\\max_j \\Delta_j$', fontsize=7.2, labelpad=1)
ax.tick_params(labelsize=6.5, length=2)
ax.set_title(f'(b) All {N:,} correct answers, eight models', loc='left',
             fontweight='bold', fontsize=6.9, pad=4)
sx, sy = dt, case['maxdj']
ax.plot([sx], [sy], marker='*', ms=10, color='black', mfc='#FFD400', mew=.6, zorder=8)
ax.annotate('case (a)', xy=(sx, sy), xytext=(9.2, 5.6), textcoords='data', fontsize=6.5,
            ha='center', va='center', zorder=8,
            arrowprops=dict(arrowstyle='-', color='black', lw=.5),
            bbox=dict(fc='white', ec='none', alpha=.8, pad=.9))
# emit both formats: main.tex includes the PDF, so writing only the PNG would
# leave a stale vector figure on disk
# A fixed canvas, not bbox_inches='tight': tight grew the canvas from
# 155.5pt to 179.7pt tall, and those extra 24pt pushed the paper to six
# pages.  Instead check_figure_layout.py is run after every regeneration
# and the margins below leave room for the (b) title inside the canvas.
# Guard against silent off-canvas clipping -- the failure mode that hid panel
# (b)'s entire x-axis.  matplotlib does not warn: it simply never emits an
# artist placed outside the figure, so the PDF is missing the ink and
# check_figure_layout.py has nothing to measure.  Every text artist that must be
# visible is therefore asserted to lie inside the canvas here, at build time.
fig.canvas.draw()
_r = fig.canvas.get_renderer()
_W, _H = fig.canvas.get_width_height()   # pixels at fig.dpi
_inside = []
# ax.title is NOT the whole story: set_title(..., loc='left') stores its artist
# in ax._left_title, leaving ax.title empty.  The `if not _a.get_text()`
# skip below then silently dropped the (b) title from this check, so a title
# wider than the canvas was clipped by matplotlib without any warning.  All
# three title artists are listed explicitly.
for _a in (list(ax.get_xticklabels()) + list(ax.get_yticklabels()) +
           [ax.xaxis.label, ax.yaxis.label, ax.title,
            ax._left_title, ax._right_title,
            t_title, t_form] + list(row_lab)):
    if not _a.get_text():
        continue
    _b = _a.get_window_extent(_r)
    _inside.append((_a.get_text(), _b.x0 >= -0.5, _b.x1 <= _W + 0.5,
                    _b.y0 >= -0.5, _b.y1 <= _H + 0.5))
_bad = [t for t in _inside if not all(t[1:])]
if _bad:
    raise SystemExit('figure text outside canvas (would be silently dropped): %r' % (_bad,))
# the x-axis must actually carry ticks and a label, not merely have been asked to
_ntx = sum(1 for t in ax.get_xticklabels() if t.get_text())
if _ntx == 0 or not ax.xaxis.label.get_text():
    raise SystemExit('panel (b) x-axis has no tick labels / label')
print('x-axis guard OK: %d tick labels + label inside canvas' % _ntx)

fig.savefig(G / 'attribution_overview.pdf'); fig.savefig(G / 'attribution_overview.png', dpi=300)
plt.close(fig)
print('case mf/mt/mj', mf, mt, mj, 'dt', round(dt, 3), 'dj', round(dj, 3), 'H', round(dt - max(0, dj), 3))
print('correct', N, 'fail', len(fail), 'cons', len(cons), 'ung', len(ung), 'conf', len(con))
