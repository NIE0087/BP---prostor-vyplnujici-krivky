#will delete this file later

import matplotlib.pyplot as plt

def draw_fixed_grids():
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # ========== M=1 ==========
    axs[0].set_xlim(-2, 2)
    axs[0].set_ylim(-2, 2)
    axs[0].axhline(0, color='black', linewidth=1.2)
    axs[0].axvline(0, color='black', linewidth=1.2)

    # čáry pro rozdělení na 2x2
    axs[0].plot([0, 0], [-2, 2], color="black", linewidth=1.2)
    axs[0].plot([-2, 2], [0, 0], color="black", linewidth=1.2)

    # popisky doprostřed čtverců
    axs[0].text(-1, -1, "D(0)", fontsize=12, ha='center', va='center')
    axs[0].text( 1, -1, "D(1)", fontsize=12, ha='center', va='center')
    axs[0].text( 1,  1, "D(2)", fontsize=12, ha='center', va='center')
    axs[0].text(-1,  1, "D(3)", fontsize=12, ha='center', va='center')

    axs[0].set_title("n=1", fontsize=14)

    # ========== M=2 ==========
    axs[1].set_xlim(-2, 2)
    axs[1].set_ylim(-2, 2)
    axs[1].axhline(0, color='black', linewidth=1.2)
    axs[1].axvline(0, color='black', linewidth=1.2)

    # mřížka rozdělení (4x4)
    for x in [-1, 0, 1]:
        axs[1].plot([x, x], [-2, 2], color="black", linewidth=1.0)
    for y in [-1, 0, 1]:
        axs[1].plot([-2, 2], [y, y], color="black", linewidth=1.0)

    # středy buněk
    xs = [-1.5, -0.5, 0.5, 1.5]
    ys = [-1.5, -0.5, 0.5, 1.5]
    coords = [(x, y) for y in ys for x in xs]

    labels = [
        "D(0,0)", "D(0,3)", "D(1,0)", "D(1,1)",
        "D(0,1)", "D(0,2)", "D(1,3)", "D(1,2)",
        "D(3,2)", "D(3,1)", "D(2,0)", "D(2,1)",
        "D(3,3)", "D(3,0)", "D(2,3)", "D(2,2)"
    ]

    for (x, y), label in zip(coords, labels):
        axs[1].text(x, y, label, fontsize=10, ha='center', va='center')

    axs[1].set_title("n=2", fontsize=14)

    # skrytí os
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.tight_layout()
    return fig

# vykreslení a uložení
fig = draw_fixed_grids()
fig.savefig("mrizky_cernobile.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Obrázek byl uložen jako mrizky_cernobile.png")
def draw_intervals():
    fig, ax = plt.subplots(figsize=(12, 2))

    # osa
    ax.hlines(0, 0, 1, color="black")
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlim(-0.05, 1.05)

    # hlavní intervaly d(0)...d(3)
    main_intervals = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)]
    for i, (start, end) in enumerate(main_intervals):
        ax.annotate(f"d({i})", xy=((start+end)/2, 0.8),
                    ha="center", va="center", fontsize=12)
        ax.plot([start, end], [0.6, 0.6], color="black")
        if i < 3:
            ax.plot([end, end], [0.6, 0.45], color="black")

    # malé intervaly d(i,j)
    small_intervals = []
    for i, (start, end) in enumerate(main_intervals):
        step = (end - start) / 4
        for j in range(4):
            s = start + j * step
            e = s + step
            ax.annotate(f"d({i},{j})", xy=((s+e)/2, -0.2),
                        ha="center", va="center", fontsize=10)
            # závorky
            if j == 0:
                ax.text(s, 0, "[", ha="center", va="center", fontsize=12)
                ax.text(s-0.004, 0, ")", ha="center", va="center", fontsize=12)
            else:
                ax.text(s-0.004, 0, ")", ha="center", va="center", fontsize=12)
                ax.text(s, 0, "[", ha="center", va="center", fontsize=12)
            if i == 3 and j == 3:
                ax.text(e, 0, "]", ha="center", va="center", fontsize=12)

    # popisky os
    ax.text(0, -0.4, "0", ha="center", va="center", fontsize=12)
    ax.text(1, -0.4, "1", ha="center", va="center", fontsize=12)
    ax.text(1.05, 0, "x", ha="left", va="center", fontsize=12)

    ax.axis("off")
    plt.tight_layout()
    return fig

fig = draw_intervals()
fig.savefig("intervaly.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Obrázek byl uložen jako intervaly.png")

fig, ax = plt.subplots(figsize=(5,5))

# Velký čtverec (0,0)-(1,1)
ax.plot([0,1,1,0,0],[0,0,1,1,0], color="black")

# Rozdělení na poloviny
ax.axhline(0.5, color="black")
ax.axvline(0.5, color="black")

# Vnořená mřížka ve spodním levém čtverci
ax.plot([0,0.5,0.5,0,0],[0,0,0.5,0.5,0], color="black")
ax.axhline(0.25, xmin=0, xmax=0.5, color="black")
ax.axvline(0.25, ymin=0, ymax=0.5, color="black")

# Ještě jedna úroveň v levém dolním kvadrantu té malé mřížky
ax.plot([0,0.25,0.25,0,0],[0,0,0.25,0.25,0], color="black")
ax.axhline(0.125, xmin=0, xmax=0.25, color="black")
ax.axvline(0.125, ymin=0, ymax=0.25, color="black")

# Osy a popisky
ax.set_xticks([0,0.25,0.5,0.75,1])
ax.set_yticks([0,0.25,0.5,0.75,1])
ax.set_xticklabels(["0","1/4","1/2","3/4","1"])
ax.set_yticklabels(["0","1/4","1/2","3/4","1"])

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_aspect("equal")
ax.axis("on")

plt.show()