"""Palette generation.

This script aims to generate a personal color palette with distinguishable
colors, which are also distinct when converted to grayscale.
"""

from typing import Iterable, List
from colorspacious import cspace_convert
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


def clipped(rgb: sRGBColor) -> sRGBColor:
    """Clip an RGB color within the [0, 1] range"""
    rgb.rgb_r = np.clip(rgb.rgb_r, 0, 1)
    rgb.rgb_g = np.clip(rgb.rgb_g, 0, 1)
    rgb.rgb_b = np.clip(rgb.rgb_b, 0, 1)
    return rgb


class Palette(List[LabColor]):
    """A palette of colors."""

    colors: List[LabColor]

    def __init__(self, colors: Iterable[str]):
        self.colors = []
        for c in colors:
            rgb = sRGBColor.new_from_rgb_hex(c)
            self.colors.append(convert_color(rgb, LabColor))

    def optimize(self):
        """Optimize the colors in terms of luminance."""
        # Adjust luminance.
        for i, c in enumerate(self.colors):
            c.lab_l = 80 / len(self.colors) * (i + 1)

        # Shuffle them.
        # n = len(self.colors)
        # lower = self.colors[:n // 2]
        # upper = self.colors[n // 2:]
        # self.colors = []
        # while len(lower) != 0 or len(upper) != 0:
        #     if len(lower) != 0:
        #         self.colors.append(lower.pop())
        #     if len(upper) != 0:
        #         self.colors.append(upper.pop())

    def show(self, save: bool = False):
        """Showcase the palette."""
        plt.rcParams["figure.dpi"] = 200
        plt.rcParams["savefig.dpi"] = 300

        # Compute coordinates.
        dx = 1
        x = np.arange(dx / 2, dx * len(self.colors), dx)
        meshx = np.arange(0, dx * (len(self.colors) + 0.5), dx)
        heights = np.asarray([0.8, 0.5, 0.3, 0.3, 0.3])
        gap = 0.1
        y = np.asarray(
            [-sum(heights[:i]) - heights[i] * 0.5 - gap * i for i in range(5)]
        )
        meshy = np.asarray(
            [[y[i] - heights[i] / 2, y[i] + heights[i] / 2] for i in range(5)]
        )
        width = dx * len(self.colors) + 1
        height = heights.sum() + gap * 5
        fig, _ = plt.subplots(figsize=(width, height), layout="tight")

        # Normal colors.
        normal_row = np.zeros((len(self.colors), 3))
        for i, color in enumerate(self.colors):
            rgb: sRGBColor = clipped(convert_color(color, sRGBColor))
            normal_row[i, 0] = rgb.rgb_r
            normal_row[i, 1] = rgb.rgb_g
            normal_row[i, 2] = rgb.rgb_b
            plt.annotate(
                rgb.get_rgb_hex(),
                (x[i], y[0]),
                fontsize=10,
                color="white",
                horizontalalignment="center",
                verticalalignment="center",
            )

        # Grayscale colors.
        gs_row = np.zeros((len(self.colors), 3))
        for i, color in enumerate(self.colors):
            rgb: sRGBColor = convert_color(color, sRGBColor)
            gs_row[i, 0] = color.lab_l / 100
            gs_row[i, 1] = color.lab_l / 100
            gs_row[i, 2] = color.lab_l / 100
            plt.annotate(
                "$L^* = %d$" % int(np.round(color.lab_l)),
                (x[i], y[1]),
                fontsize=10,
                color="white",
                horizontalalignment="center",
                verticalalignment="center",
            )

        # Color blindness simulators.
        deu_space = dict(name="sRGB1+CVD", cvd_type="deuteranomaly", severity=80)
        deu_row = cspace_convert(normal_row, deu_space, "sRGB1")
        pro_space = dict(name="sRGB1+CVD", cvd_type="protanomaly", severity=80)
        pro_row = cspace_convert(normal_row, pro_space, "sRGB1")
        tri_space = dict(name="sRGB1+CVD", cvd_type="tritanomaly", severity=80)
        tri_row = cspace_convert(normal_row, tri_space, "sRGB1")

        # Plot the rows.
        rows = [normal_row, gs_row, deu_row, pro_row, tri_row]
        labels = ["Normal", "Grayscale", "Deuteranopia", "Protanopia", "Tritanopia"]
        for i in range(5):
            colors = np.clip([rows[i]], 0, 1)
            plt.pcolormesh(meshx, meshy[i], colors)
            plt.annotate(
                labels[i],
                (0, y[i]),
                (-10, 0),
                textcoords="offset points",
                fontsize=10,
                color="black",
                horizontalalignment="right",
                verticalalignment="center",
            )

        # Show the palettes.
        plt.axis("equal")
        plt.axis("off")
        plt.yticks(0.5 * y, labels=labels)
        plt.show()
        if save:
            fig.savefig("../images/palette.png")


if __name__ == "__main__":
    colors = [
        ("RoyalBlue", "#4169e1"),
        ("OrangeRed", "#ff4500"),
        ("BlueViolet", "#8a2be2"),
        ("SeaGreen", "#2e8b57"),
        ("HotPink", "#ff69b4"),
        ("GoldenRod", "#daa520"),
    ]
    # palette = Palette(["#5ba0fa", "#ff0000", "#ad5ff5", "#0cf01f", "#f0305f", "#daa520"])
    palette = Palette(map(lambda x: x[1], colors))
    palette.optimize()
    palette.show(save=True)
