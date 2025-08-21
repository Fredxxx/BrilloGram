# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 11:57:02 2025

@author: Goerlitz
"""

                
import os
import matplotlib.pyplot as plt

master_folder = "C:\\Users\\Goerlitz\\Documents\\temp\\20250728_NAchange\\"

# for folder_name in os.listdir(master_folder):
#     folder_path = os.path.join(master_folder, folder_name)
#     if os.path.isdir(folder_path):
#         # Loop through files in this folder
#         for file_name in os.listdir(folder_path):
#             if file_name.endswith('.txt'):
#                 print(file_name)
#                 file_path = os.path.join(folder_path, file_name)
#                 # with open(file_path, 'r') as file:
#                 #     content = file.read()
#                 #     print(f"Content of {file_path}:\n{content}\n")
                



time_values = None  # To store the common time values
theta_data = []     # List of tuples: (label, theta_values)
phi_data = []       # List of tuples: (label, phi_values)

for folder_name in os.listdir(master_folder):
    folder_path = os.path.join(master_folder, folder_name)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            print(file_name)
            if file_name.endswith('.txt'):
                print(file_name)
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r') as f:
                    times = []
                    thetas = []
                    phis = []
                    for line in f:
                        if line.strip():
                            cols = line.split()
                            times.append(float(cols[0]))
                            thetas.append(float(cols[1]))
                            phis.append(float(cols[2]))
                    if time_values is None:
                        time_values = times
                    else:
                        # Check if time values are consistent
                        if times != time_values:
                            print(f"Warning: Time column mismatch in {file_path}")
                    label = f"{folder_name}/{file_name}"
                    theta_data.append((label, thetas))
                    phi_data.append((label, phis))

# Plot Theta vs Time
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for label, theta in theta_data:
    plt.plot(time_values, theta, label=label)
plt.title('Theta (deg) vs Time')
plt.xlabel('Time')
plt.ylabel('Theta (deg)')
plt.legend(fontsize='small', loc='best')

# Plot Phi vs Time
plt.subplot(1, 2, 2)
for label, phi in phi_data:
    plt.plot(time_values, phi, label=label)
plt.title('Phi (deg) vs Time')
plt.xlabel('Time')
plt.ylabel('Phi (deg)')
plt.legend(fontsize='small', loc='best')

plt.tight_layout()
plt.show()