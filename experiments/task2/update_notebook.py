
import json
import os

nb_path = 'experiments/task2/explore.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

# The code to inject
source_code = [
    "# --- Visualize Conv1 Kernels ---\n",
    "# conv1 weights shape: (out_channels, in_channels, kH, kW) = (8, 3, 5, 5)\n",
    "kernels = model.conv1.weight.detach().cpu()\n",
    "print(\"Conv1 kernels shape:\", kernels.shape)\n",
    "\n",
    "fig, axes = plt.subplots(1, 8, figsize=(12, 3))\n",
    "for i, ax in enumerate(axes):\n",
    "    # Normalize to [0, 1] for display\n",
    "    k = kernels[i] # (3, 5, 5)\n",
    "    k = k - k.min()\n",
    "    k = k / k.max()\n",
    "    k = k.permute(1, 2, 0) # (5, 5, 3)\n",
    "    ax.imshow(k)\n",
    "    ax.axis(\"off\")\n",
    "    ax.set_title(f\"Filter {i}\")\n",
    "plt.show()"
]

# Find the empty code cell at the end (id: 33c7307a)
# Or just append if we want to be safe.
# The user asked to "add a cell".
# The notebook currently has an empty code cell at the end. I will fill it.

cells = nb['cells']
target_id = "33c7307a"
found = False

for cell in cells:
    if cell.get('id') == target_id:
        cell['source'] = source_code
        found = True
        break

if not found:
    # Append if not found
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "conv1_viz",
        "metadata": {},
        "outputs": [],
        "source": source_code
    }
    cells.append(new_cell)

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Updated explore.ipynb")
