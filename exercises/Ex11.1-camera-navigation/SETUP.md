# Ex_11.1 — Setup Guide

**Print this and keep it on the bench.** Everything here happens before any
machine learning.

---

## 0 · What you need

Per team: one JetRacer Pro, one microSD card (**≥ 64 GB**), a card reader, four
charged 18650 cells, a laptop, and the WiFi credentials.

**Label your car and your SD card with the same number.** Six identical cars and
six SD cards will otherwise be mixed up within one session, and a mixed-up card
costs you your dataset.

---

## 1 · Flash the SD card

1. Download the **Waveshare JetRacer image** from the JetRacer Pro AI Kit wiki.
   It is pre-built on JetPack 4.5. **Do not build your own image** and do not
   use an NVIDIA JetBot image — see the trap below.
2. Unzip it.
3. Write it to the card with **Etcher**.
4. Eject cleanly.

Allow 20–40 minutes. Start the download before the session if you can.

---

## 2 · First boot

1. Insert the card (the slot is on the **back** of the Nano module).
2. Fit charged cells. Check polarity twice.
3. Power on and wait.
4. The **OLED shows the car's IP address** once it joins the WiFi.

No IP after two minutes: check the WiFi configuration, then the card seating.

---

## 3 · Connect

Open `http://<ip>:8888` in a browser. JupyterLab is running **on the car** —
you write code in your browser and it executes on the Nano.

---

## 4 · The two traps

### Trap 1 — Waveshare code, not NVIDIA's

The motor drive code differs between the Waveshare and NVIDIA projects. If you
follow an NVIDIA JetRacer tutorial and overwrite the `jetracer` folder, **the
car will not drive**. Remove the folder and reinstall the Waveshare version.

Symptom: everything imports, no error, wheels do not turn.

### Trap 2 — 5 W power mode

The Nano must be placed in 5 W mode so it does not draw more current than the
battery pack can supply. Skip this and the car **resets under load** — usually
mid-demonstration. Set it in a terminal before you start, and confirm after
every reboot.

---

## 5 · Confirm it drives

Open `/jetracer/notebooks/` and run **`basic_motion.ipynb`**.

If the wheels turn and the steering moves, the stack works. **Do not proceed
until this passes.** Debugging a neural network on a car that was never driving
is a wasted afternoon.

---

## 6 · Then, and only then

Open `Ex_11.1_00_hardware_check.ipynb` from this exercise.

---

## Safety

- **One person drives, another watches with a hand on the stop.**
- Keep `throttle_gain` at 0.4 or below until the loop rate is measured.
- Lithium cells: supplied charger only, never unattended, never run flat.
  The kit's warning is not boilerplate.

## Note on hardware

JetRacer Pro does not support the Orin series — different power architecture,
incompatible project. These cars are what they are.
