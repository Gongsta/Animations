from manim import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import distance_transform_edt
from skimage import morphology as morph
from skimage.morphology import skeletonize

# config.disable_caching = True


class Euc(ZoomedScene):
    def construct(self):
        # Create a small image
        bw = np.zeros((9, 9))
        bw[4, 4] = 255
        bw = morph.dilation(bw, morph.disk(3))
        manim_bw = ImageMobject(bw)
        manim_bw.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        manim_bw.height = 7

        text = Tex("Euclidean Distance Transform", font_size=60)

        self.play(Write(text), run_time=2)
        self.wait(4)

        self.play(GrowFromCenter(manim_bw), ShrinkToCenter(text), run_time=2)
        self.wait(2)

        # Compute the distance transform
        im_dist = distance_transform_edt(bw)
        im_dist_upscaled = im_dist.copy()
        im_dist_upscaled *= 80
        manim_im_dist = ImageMobject(im_dist_upscaled)
        manim_im_dist.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        manim_im_dist.height = 7

        text_array = []
        for (i, j), z in np.ndenumerate(im_dist):
            if z == 0.0:
                text_array.append(
                    Tex(str(round(z, 2)), color=WHITE, font_size=20).move_to(
                        (-4 + j) * 0.78 * RIGHT + (-4 + i) * 0.78 * DOWN
                    )
                )
            else:
                text_array.append(
                    Tex(str(round(z, 2)), color=BLACK, font_size=20).move_to(
                        (-4 + j) * 0.78 * RIGHT + (-4 + i) * 0.78 * DOWN
                    )
                )

        text_array_group = VGroup(*text_array)
        self.play(Transform(manim_bw, manim_im_dist), FadeIn(text_array_group), run_time=2)
        self.wait(2)


class DoYouSee(ZoomedScene):
    def construct(self):
        text1 = Tex("Do you see how this can be useful", font_size=40)
        text2 = Tex("for extracting the centerline?", font_size=40).next_to(text1, DOWN)
        self.play(Write(text1))
        self.play(Write(text2))
        self.wait(2)


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


class Track2(ZoomedScene):
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
        map_img_path = f"../maps/{MAP_NAME}.pgm"
        map_yaml_path = f"../maps/{MAP_NAME}.yaml"
        raw_map_img = np.array(Image.open(map_img_path))
        raw_map_img = raw_map_img.astype(np.float64)
        print(raw_map_img)

        manim_raw_map_img = ImageMobject(raw_map_img).shift(DOWN)
        manim_raw_map_img.height = 9

        # grayscale -> binary. Converts grey to black
        map_img = raw_map_img.copy()
        map_img[map_img <= 210.0] = 0
        map_img[map_img > 210.0] = 255

        map_img = map_img.astype(np.uint8)
        map_height = map_img.shape[0]

        manim_map_img = ImageMobject(map_img).shift(DOWN)
        manim_map_img.height = 9

        # Get the euclidean distance transofrm
        dist_transform = scipy.ndimage.distance_transform_edt(map_img)
        dist_transform_upscaled = dist_transform.copy()
        dist_transform_upscaled *= 5
        manim_dist_transform = ImageMobject(dist_transform_upscaled).shift(DOWN)
        manim_dist_transform.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        manim_dist_transform.height = 9

        ## Skeletonize the centerline
        THRESHOLD = 0.17  # You should play around with this number. Is you say hairy lines generated, either clean the map so it is more curvy or increase this number
        centers = dist_transform > THRESHOLD * dist_transform.max()
        centerline = skeletonize(centers)
        centerline_upscaled = centerline.copy().astype(np.uint8) * 255
        manim_centerline = ImageMobject(centerline_upscaled).shift(DOWN)
        manim_centerline.height = 9
        self.play(FadeIn(manim_centerline))
        # self.play(FadeIn(manim_centerline))
        self.wait(2)

        self.play(FadeIn(manim_map_img))
