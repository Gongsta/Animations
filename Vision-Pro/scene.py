from manim import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import distance_transform_edt
from skimage import morphology as morph
from skimage.morphology import skeletonize

config.background_color = "#1f1f1f"


class Track(ZoomedScene):
    # contributed by TheoremofBeethoven, www.youtube.com/c/TheoremofBeethoven
    def __init__(self, **kwargs):
        ZoomedScene.__init__(
            self,
            zoom_factor=0.3,
            zoomed_display_height=2,
            zoomed_display_width=2.5,
            image_frame_stroke_width=10,
            zoomed_camera_config={
                "default_frame_stroke_width": 3,
            },
            **kwargs,
        )

    def construct(self):
        MAP_NAME = "e7_floor5_large"
        map_img_path = f"maps/{MAP_NAME}.pgm"
        map_yaml_path = f"maps/{MAP_NAME}.yaml"
        raw_map_img = np.array(Image.open(map_img_path))
        raw_map_img = raw_map_img.astype(np.float64)
        print(raw_map_img)

        manim_raw_map_img = ImageMobject(raw_map_img).shift(DOWN)
        manim_raw_map_img.height = 9
        self.play(FadeIn(manim_raw_map_img))

        self.wait(2)
        # grayscale -> binary. Converts grey to black
        map_img = raw_map_img.copy()
        map_img[map_img <= 210.0] = 0
        map_img[map_img > 210.0] = 255

        map_img = map_img.astype(np.uint8)
        map_height = map_img.shape[0]

        manim_map_img = ImageMobject(map_img).shift(DOWN)
        manim_map_img.height = 9
        self.play(Transform(manim_raw_map_img, manim_map_img))
        self.wait(2)

        # Get the euclidean distance transofrm
        dist_transform = scipy.ndimage.distance_transform_edt(map_img)
        dist_transform_upscaled = dist_transform.copy()
        dist_transform_upscaled *= 5
        manim_dist_transform = ImageMobject(dist_transform_upscaled).shift(DOWN)
        manim_dist_transform.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        manim_dist_transform.height = 9
        self.play(Transform(manim_map_img, manim_dist_transform))

        self.wait(2)

        zoomed_camera = self.zoomed_camera
        zoomed_display = self.zoomed_display
        frame = zoomed_camera.frame
        zoomed_display_frame = zoomed_display.display_frame
        ScaleInPlace(zoomed_display, 2)
        zoomed_display.move_to(4 * LEFT)

        frame.shift(1.2 * UP)

        zd_rect = BackgroundRectangle(zoomed_display, fill_opacity=0, buff=MED_SMALL_BUFF)
        self.add_foreground_mobject(zd_rect)

        unfold_camera = UpdateFromFunc(zd_rect, lambda rect: rect.replace(zoomed_display))

        self.play(Create(frame))
        self.activate_zooming()

        self.play(self.get_zoomed_display_pop_out_animation(), unfold_camera, run_time=1.5)

        # zoomed_camera_text = Tex("Euclidean Distance", font_size=30)
        # zoomed_camera_text.next_to(zoomed_display_frame, DOWN)
        # self.play(FadeIn(zoomed_camera_text, shift=UP))
        # self.wait()

        # self.play(
        # 	FadeOut(zoomed_camera_text),
        # # )

        ## Skeletonize the centerline
        THRESHOLD = 0.17  # You should play around with this number. Is you say hairy lines generated, either clean the map so it is more curvy or increase this number
        centers = dist_transform > THRESHOLD * dist_transform.max()
        centerline = skeletonize(centers)
        centerline_upscaled = centerline.copy().astype(np.uint8) * 255
        manim_centerline = ImageMobject(centerline_upscaled).shift(DOWN)
        manim_centerline.height = 9
        self.play(
            AnimationGroup(
                frame.animate.shift(1.5 * RIGHT + 0.5 * DOWN),
                Transform(manim_dist_transform, manim_centerline),
                lag_ratio=0.2,
            ),
            run_time=4,
        )
        self.wait()
        # self.play(FadeIn(manim_centerline))

        # Exit the zooming
        self.play(
            self.get_zoomed_display_pop_out_animation(),
            unfold_camera,
            rate_func=lambda t: smooth(1 - t),
        )
        self.play(Uncreate(zoomed_display_frame), FadeOut(frame))
        self.wait(2)





class DisparityEq(Scene):
    def construct(self):
        t3 = MathTex(r"Z").scale(2)
        t2 = MathTex(r"\propto \frac{1}{x_l - x_r}").scale(2)
        t = MathTex(r"= \frac{f \cdot T}{x_l - x_r}").scale(2)
        box = Rectangle(width=3.5, height=1.1, color=BLUE)
        box.move_to(t[0][7])

        focal_length = Tex("Focal Length").shift(UP * 2 + LEFT * 1)
        f_line = Line(
            start=focal_length.get_bottom() + 0.2 * DOWN, end=t[0][1].get_left() + 0.2 * LEFT
        )
        baseline = Tex("Baseline").shift(UP * 2 + RIGHT * 3.5)
        b_line = Line(
            start=baseline.get_bottom() + 0.2 * DOWN + 0.2 * LEFT,
            end=t[0][3].get_right() + 0.2 * RIGHT,
        )
        depth = Tex("Depth").shift(DOWN * 2 + LEFT * 3.5)
        d_line = Line(start=t3.get_bottom() + LEFT * 3 + 0.2 * DOWN, end=depth.get_top() + 0.2 * UP)
        disparity = Tex("Disparity").shift(DOWN * 2 + RIGHT * 3.5)
        dis_line = Line(start=box.get_bottom() + 0.2 * RIGHT, end=disparity.get_left() + 0.2 * LEFT)

        self.play(Write(t3))
        self.play(AnimationGroup(t3.animate.shift(LEFT * 3), Write(t2), lag_ratio=0.5))
        self.play(
            AnimationGroup(Write(box), FadeIn(dis_line, d_line, disparity, depth), lag_ratio=0.1)
        )
        self.wait(0.5)
        # self.play(ReplacementTransform(t3, t2))
        self.play(
            AnimationGroup(
                ReplacementTransform(t2, t),
                FadeIn(focal_length, baseline, f_line, b_line),
                lag_ratio=0.5,
            )
        )
        self.wait(2)


class Camera(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        circle = Circle()

        video = ImageMobject(raw_map_img).shift(DOWN)
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(circle, axes)
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait()
        self.stop_ambient_camera_rotation()
        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES)
        self.wait()
