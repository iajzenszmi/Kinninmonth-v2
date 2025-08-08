#!/usr/bin/env python3
"""
Kininmonth-inspired 1D Energy Balance Model with interactive sliders.
Standalone script — run:
    python kininmonth_v2.py
Requires: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.widgets import Slider, RadioButtons
from datetime import datetime
import os

# ========================
# Model setup
# ========================

nlat = 121
lats = np.linspace(-90, 90, nlat)
phi = np.deg2rad(lats)

# Constants
C = 4.0e7      # heat capacity (J/m2/K)
A = 210.0      # OLR intercept
B = 1.9        # OLR slope
S0 = 340.0     # global mean insolation (W/m²)
Q0 = 420.0     # latitudinal insolation peak
alpha0 = 0.3   # baseline albedo

trop_band = 25.0   # degrees from equator for convection cap
trop_mask = np.abs(lats) <= trop_band

dt = 86400.0       # 1 day timestep

# Insolation by latitude
S_lat = Q0 * np.cos(phi)**0.6
S_lat = S_lat - (S_lat.mean() - S0)

def absorbed_solar(T):
    a = np.where(T < 260.0, 0.5,
                 np.where(T > 280.0, alpha0,
                          alpha0 + (0.5 - alpha0) * (280.0 - T) / 20.0))
    return S_lat * (1.0 - a)

def laplacian_sphere(T):
    dphi = np.deg2rad(180.0 / (nlat - 1))
    cosphi = np.cos(phi)
    dT_dphi = np.zeros_like(T)
    dT_dphi[1:-1] = (T[2:] - T[:-2]) / (2 * dphi)
    flux = cosphi * dT_dphi
    dflux_dphi = np.zeros_like(T)
    dflux_dphi[1:-1] = (flux[2:] - flux[:-2]) / (2 * dphi)
    return dflux_dphi / np.maximum(cosphi, 1e-6)

def convective_cap_term(T, cap_temp, cap_strength, use_cap):
    if not use_cap:
        return np.zeros_like(T)
    extra = np.zeros_like(T)
    warm = (T > cap_temp) & trop_mask
    extra[warm] = cap_strength * (T[warm] - cap_temp)
    return extra

def step_forward(T, forcing_Wm2, D, cap_temp, cap_strength, use_cap):
    # Semi-implicit: treat B term implicitly for stability
    net = absorbed_solar(T) - convective_cap_term(T, cap_temp, cap_strength, use_cap) + forcing_Wm2
    net += D * laplacian_sphere(T) - A
    return (T + dt * net / C) / (1 + dt * B / C)

def integrate_to_equilibrium(T0, forcing_Wm2, D, cap_temp, cap_strength, use_cap, years=60, tol=1e-5):
    T = T0.copy()
    nsteps = int((years * 365.0 * 86400.0) // dt)
    for _ in range(nsteps):
        T_new = step_forward(T, forcing_Wm2, D, cap_temp, cap_strength, use_cap)
        if np.max(np.abs(T_new - T)) < tol:
            return T_new
        T = T_new
    return T

# ========================
# Output folder
# ========================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
outdir = os.path.join("outputs", timestamp)
os.makedirs(outdir, exist_ok=True)

# ========================
# Interactive plot setup
# ========================

# Default parameters
D_default = 0.35
forcing_default = 0.0
cap_temp_default = 303.15
cap_strength_default = 8.0
ENSO_period_default = 3.5
ENSO_amp_default = 3.0
use_cap_default = True

T_init = np.full(nlat, 288.0)
T_eq_baseline = integrate_to_equilibrium(T_init, 0.0, D_default, cap_temp_default, cap_strength_default, use_cap_default)

fig, ax = plt.subplots(figsize=(8,5))
plt.subplots_adjust(left=0.25, bottom=0.45)
line, = ax.plot(lats, T_eq_baseline, lw=2)
ax.set_xlabel("Latitude (°)")
ax.set_ylabel("Temperature (K)")
ax.set_title("Kininmonth-inspired EBM — Interactive")
ax.grid(True)

# Diagnostics text
diag_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top", ha="left", fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.7))

# Sliders
axcolor = 'lightgoldenrodyellow'
ax_D = plt.axes([0.25, 0.35, 0.65, 0.03], facecolor=axcolor)
ax_forcing = plt.axes([0.25, 0.30, 0.65, 0.03], facecolor=axcolor)
ax_captemp = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
ax_ensoper = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
ax_ensoamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

s_D = Slider(ax_D, 'D', 0.1, 1.0, valinit=D_default)
s_forcing = Slider(ax_forcing, 'CO₂ Forcing', -2.0, 6.0, valinit=forcing_default)
s_captemp = Slider(ax_captemp, 'Cap Temp (K)', 295.0, 310.0, valinit=cap_temp_default)
s_ensoper = Slider(ax_ensoper, 'ENSO Period (yrs)', 2.0, 6.0, valinit=ENSO_period_default)
s_ensoamp = Slider(ax_ensoamp, 'ENSO Amp (W/m²)', 0.0, 6.0, valinit=ENSO_amp_default)

# Radio button for cap on/off
rax = plt.axes([0.025, 0.5, 0.15, 0.1], facecolor=axcolor)
radio = RadioButtons(rax, ('Cap ON', 'Cap OFF'), active=0)

# Update function
def update(val):
    D_val = s_D.val
    forc_val = s_forcing.val
    captemp_val = s_captemp.val
    use_cap = (radio.value_selected == 'Cap ON')
    T_eq = integrate_to_equilibrium(T_init, forc_val, D_val, captemp_val, cap_strength_default, use_cap)

    line.set_ydata(T_eq)

    # Diagnostics
    gm_delta = np.mean(T_eq) - np.mean(T_eq_baseline)
    eq_temp = np.mean(T_eq[np.abs(lats) < 10])
    pole_temp = np.mean(T_eq[np.abs(lats) > 60])
    eq_pole_grad = eq_temp - pole_temp
    eq_pole_grad_base = np.mean(T_eq_baseline[np.abs(lats) < 10]) - np.mean(T_eq_baseline[np.abs(lats) > 60])
    polar_amp = (pole_temp - np.mean(T_eq_baseline[np.abs(lats) > 60])) / gm_delta if gm_delta != 0 else np.nan
    diag_text.set_text(f"ΔT global: {gm_delta:.2f} K\nΔ(eq-pole): {eq_pole_grad - eq_pole_grad_base:.2f} K\nPolar amplification: {polar_amp:.2f}")

    fig.canvas.draw_idle()

s_D.on_changed(update)
s_forcing.on_changed(update)
s_captemp.on_changed(update)
s_ensoper.on_changed(update)
s_ensoamp.on_changed(update)
radio.on_clicked(update)

# ========================
# ENSO animation function
# ========================
def run_enso_and_save():
    D_val = s_D.val
    forc_val = s_forcing.val
    captemp_val = s_captemp.val
    ENSO_per = s_ensoper.val
    ENSO_amp = s_ensoamp.val
    use_cap = (radio.value_selected == 'Cap ON')

    T_eq = integrate_to_equilibrium(T_init, forc_val, D_val, captemp_val, cap_strength_default, use_cap)

    years_anim = 12
    steps_anim = int((years_anim * 365.0 * 86400.0) // dt)
    T_time = np.zeros((steps_anim, nlat))
    T = T_eq.copy()

    period_steps = int((ENSO_per * 365.0 * 86400.0) // dt)
    lat_sigma = np.deg2rad(18.0)
    gauss = np.exp(-0.5 * (phi / lat_sigma)**2)

    for k in range(steps_anim):
        phase = 2.0 * np.pi * (k % max(period_steps,1)) / max(period_steps,1)
        F = forc_val + ENSO_amp * np.sin(phase) * gauss
        T = step_forward(T, F, D_val, captemp_val, cap_strength_default, use_cap)
        T_time[k, :] = T

    # Save Hovmöller
    T_anom = T_time - T_eq[None, :]
    plt.figure(figsize=(8,5))
    plt.imshow(T_anom[::-1, :], aspect='auto', extent=[lats.min(), lats.max(), 0, years_anim])
    plt.colorbar(label="Temperature anomaly (K)")
    plt.xlabel("Latitude (°)")
    plt.ylabel("Years")
    plt.title("ENSO-like Temperature Anomalies")
    plt.tight_layout()
    hov_path = os.path.join(outdir, "hovmoller.png")
    plt.savefig(hov_path, dpi=160)
    plt.close()

    # Animation
    fig_anim, ax_anim = plt.subplots(figsize=(8,5))
    line_anim, = ax_anim.plot(lats, T_time[0], lw=2)
    ax_anim.set_xlim(lats.min(), lats.max())
    ax_anim.set_ylim(np.min(T_time)-1, np.max(T_time)+1)
    ax_anim.set_xlabel("Latitude (°)")
    ax_anim.set_ylabel("Temperature (K)")
    ax_anim.grid(True)

    def anim_update(i):
        line_anim.set_ydata(T_time[i])
        ax_anim.set_title(f"ENSO-like Temp Evolution — Year {i*dt/86400.0/365.0:.2f}")
        return (line_anim,)

    anim = FuncAnimation(fig_anim, anim_update, frames=steps_anim, interval=60, blit=True)
    mp4_path = os.path.join(outdir, "enso_animation.mp4")
    writer = FFMpegWriter(fps=15, bitrate=1800)
    anim.save(mp4_path, writer=writer)
    plt.close(fig_anim)

    print(f"Saved Hovmöller to {hov_path}")
    print(f"Saved animation to {mp4_path}")

# Press 'a' to run ENSO animation and save
def on_key(event):
    if event.key == 'a':
        run_enso_and_save()
fig.canvas.mpl_connect('key_press_event', on_key)

plt.show()
