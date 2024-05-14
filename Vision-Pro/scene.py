from manim import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import distance_transform_edt
from skimage import morphology as morph
from skimage.morphology import skeletonize

config.background_color = "#1f1f1f"

class DisparityEq(Scene):
    def construct(self):
        t3 = MathTex(r"Z").scale(2)
        t2 = MathTex(r"\propto \frac{1}{x_l - x_r}").scale(2)
        t = MathTex(r"= \frac{f \cdot T}{x_l - x_r}").scale(2)
        box = Rectangle(width=3.5, height=1.1, color=BLUE)
        box.move_to(t[0][7])

        focal_length = Tex("Focal Length").shift(UP * 2 + LEFT * 1)
        f_line = Line(start=focal_length.get_bottom() + 0.2 * DOWN, end=t[0][1].get_left() + 0.2 * LEFT)
        baseline = Tex("Baseline").shift(UP * 2 + RIGHT * 3.5)
        b_line = Line(start=baseline.get_bottom() + 0.2 * DOWN + 0.2 * LEFT, end=t[0][3].get_right() + 0.2 * RIGHT)
        depth = Tex("Depth").shift(DOWN * 2 + LEFT * 3.5)
        d_line = Line(start=t3.get_bottom() + LEFT*3 + 0.2 * DOWN, end=depth.get_top() + 0.2 * UP)
        disparity = Tex("Disparity").shift(DOWN * 2 + RIGHT * 3.5)
        dis_line = Line(start=box.get_bottom() + 0.2 * RIGHT, end=disparity.get_left() + 0.2 * LEFT)

        self.play(Write(t3))
        self.play(AnimationGroup(t3.animate.shift(LEFT * 3), Write(t2), lag_ratio=0.5))
        self.play(AnimationGroup(Write(box), FadeIn(dis_line, d_line, disparity,depth), lag_ratio=0.1))
        self.wait(0.5)
        # self.play(ReplacementTransform(t3, t2))
        self.play(AnimationGroup(ReplacementTransform(t2, t), FadeIn(focal_length, baseline, f_line, b_line), lag_ratio=0.5))
        self.wait(2)


class Camera(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        circle=Circle()

		video = ImageMobject(raw_map_img).shift(DOWN)
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(circle,axes)
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait()
        self.stop_ambient_camera_rotation()
        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES)
        self.wait()
