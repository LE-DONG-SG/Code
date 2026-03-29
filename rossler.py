import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -------------------------- All your original parameters are fully preserved --------------------------
# Core parameters of the Rossler system (classic chaotic parameters)
a = 0.2
b = 0.2
c = 5.7

# Differential equation parameters
dt = 0.001  # Time step
total_steps = 200000  # Slightly reduced for Streamlit speed, you can change back to 2000000

# Initial values
x, y, z = 0.1, 0.1, 0.1

# Coordinate scaling (consistent with your original code)
scale = 25
offset_x = 0
offset_y = -100

# -------------------------- Streamlit Page Settings --------------------------
st.title("Chaotic System - Rossler Attractor")
st.markdown("Ported from your original Python turtle code to Streamlit version")

# -------------------------- Solve Rossler Equations (exact same algorithm) --------------------------
with st.spinner("Calculating chaotic trajectory..."):
    x_list = [x]
    y_list = [y]

    for step in range(total_steps):
        # Your original Rossler differential equations, unchanged
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)

        # Your original Euler iteration, unchanged
        x += dx * dt
        y += dy * dt
        z += dz * dt

        x_list.append(x)
        y_list.append(y)

x_arr = np.array(x_list)
y_arr = np.array(y_list)

# Your original coordinate transformation
turtle_x = x_arr * scale + offset_x
turtle_y = y_arr * scale + offset_y

# -------------------------- Plotting --------------------------
fig, ax = plt.subplots(figsize=(10, 9), dpi=100)
ax.set_facecolor("black")
ax.plot(turtle_x, turtle_y, color="#00ffff", linewidth=0.8)

# Draw your original coordinate axes
ax.axhline(y=offset_y, color="gray", linewidth=1)
ax.axvline(x=offset_x, color="gray", linewidth=1)

ax.set_aspect('equal')
ax.axis("off")
plt.tight_layout()

# Display in Streamlit
st.pyplot(fig)