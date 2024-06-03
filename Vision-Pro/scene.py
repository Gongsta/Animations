from manim import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import distance_transform_edt
from skimage import morphology as morph
from skimage.morphology import skeletonize

config.background_color = "#2B2B2A"
# config.background_color = "#00b140" # Green screen

import numpy as np

np.random.seed(0)

# For pyramid_2d
position_list = [
    [4, 0, 0],  # bottom right of right triangle
    [3, 2, 0],  # top of right triangle
    [2, 0, 0],  # bottom right of left triangle / bottom left of right triangle
    [1, 2, 0],  # top of left triangle
    [0, 0, 0],  # bottom left of left triangle
]


class SLAMGraph(Scene):
    def construct(self):
        # Camera Poses
        DOT_RADIUS = 0.1 + DEFAULT_DOT_RADIUS
        pose_vertices = [
            Dot(point=[i - 2, 0, 0], color=RED, radius=DOT_RADIUS).set_label(f"P_{i+1}")
            for i in range(5)
        ]

        # 3D Landmarks
        landmark_vertices = [
            Dot(point=[i - 1.5, 2, 0], color=GREEN).set_label(f"L_{i+1}") for i in range(4)
        ]

        # Observations (Edges) (Reused from previous script)
        edges = []
        for i in range(10):
            start = np.random.randint(0, 5)
            end = np.random.randint(0, 4)
            line = Line(
                pose_vertices[start].get_center(),
                landmark_vertices[end].get_center(),
                color=WHITE,
                buff=0.3,
            )
            edges.append(line)

        graph_group = VGroup(*pose_vertices, *landmark_vertices, *edges)
        graph_group.move_to(ORIGIN)
        graph_group.scale(1.5)
        self.play(DrawBorderThenFill(graph_group), run_time=2.5)

        # ------- Explain vertices -------
        animate_poses = [pose.animate.scale(1.3) for pose in pose_vertices]
        animate_landmarks = [landmark.animate.scale(1.5) for landmark in landmark_vertices]
        reduce_opacity_edges = [edge.animate.set_opacity(0.2) for edge in edges]
        self.play(reduce_opacity_edges, animate_poses, animate_landmarks)
        self.wait(0.5)

        r = Rectangle(width=0.85, height=0.5, color=RED).set_fill(RED, 0.8)
        t = (
            Triangle(color=RED)
            .set_fill(RED, 0.8)
            .rotate(PI / 2)
            .scale(0.25)
            .next_to(r, RIGHT, buff=0)
        )
        cameras = [
            VGroup(r, t).copy().scale(0.5).move_to(pose_vertices[i].get_center()) for i in range(5)
        ]
        vertex_to_camera_anim = [
            ReplacementTransform(pose_vertices[i], cameras[i]) for i in range(5)
        ]

        pose_tex = MathTex(
            r"\text{Camera }\mathbf{Pose} = \text{Position + Orientation}", font_size=30
        ).shift(2.3 * DOWN)

        self.play(AnimationGroup(vertex_to_camera_anim, lag_ratio=0.4), FadeIn(pose_tex))

        shapes = [
            Polygon(*position_list, color=GREEN)
            .set_fill(GREEN, 0.8)
            .scale(0.1)
            .set_stroke(width=2)
            .move_to(landmark_vertices[i].get_center())
            for i in range(4)
        ]

        landmark_to_shapes_anim = [
            ReplacementTransform(landmark_vertices[i], shapes[i]) for i in range(4)
        ]

        landmark_tex = MathTex(
            r"\mathbf{Landmark} = \text{Feature in 3D space}", font_size=30
        ).shift(2.3 * UP)
        self.play(
            AnimationGroup(
                AnimationGroup(*landmark_to_shapes_anim), FadeIn(landmark_tex), lag_ratio=0.4
            )
        )

        # ------- Highlighting Edges -------
        self.play(edges[9].animate.set_opacity(1), FadeOut(landmark_tex, pose_tex), run_time=0.8)
        self.play(edges[2].animate.set_opacity(1), edges[9].animate.set_opacity(0.2), run_time=0.5)
        self.play(edges[6].animate.set_opacity(1), edges[2].animate.set_opacity(0.2), run_time=0.5)
        self.play(edges[1].animate.set_opacity(1), edges[6].animate.set_opacity(0.2), run_time=0.5)

        # Fade out everything but edges[8] and pose_vertices[3]
        to_fade_out = [
            v
            for v in cameras + shapes + edges
            if v != shapes[3] and v != cameras[0] and v != edges[1]
        ]
        self.play(FadeOut(*to_fade_out))

        self.play(
            AnimationGroup(
                edges[1].animate.set_opacity(0),
                AnimationGroup(
                    shapes[3].animate.scale(8).shift(0.4 * DOWN + 0.9 * LEFT).rotate(-PI / 25),
                    cameras[0].animate.rotate(PI / 6),
                ),
                lag_ratio=0.5,
            )
        )
        self.wait(2)


class Camera3D(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=-60 * DEGREES)

        image_plane = (
            Rectangle(width=1.92, height=1.08).rotate(PI / 2, axis=RIGHT).scale(1).shift(3.0 * DOWN)
        )
        pyramid_2d = (
            Polygon(*position_list, color=GREEN)
            .rotate(PI / 2, axis=RIGHT)
            .set_fill(GREEN, 0.8)
            .set_stroke(width=2)
            .scale(0.15)
            .move_to(image_plane.get_center() + 0.2 * LEFT)
        )

        # 3D pyramid
        vertex_coords = [[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0], [0, 0, 2.3]]
        faces_list = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 1, 2, 3]]
        pyramid = (
            Polyhedron(vertex_coords, faces_list)
            .set_color(GREEN)
            .set_fill(GREEN, 0.5)
            .set_stroke(width=0)
        )

        pyramid.graph.set_opacity(0)
        pyramid_left = pyramid.copy()
        pyramid_right = pyramid.copy().next_to(pyramid_left, RIGHT, buff=0)
        pyramid_3d = VGroup(pyramid_left, pyramid_right).move_to(ORIGIN + 7 * UP + 2 * LEFT)

        # Create the camera
        prism = Prism(dimensions=[0.5, 0.85, 0.5]).set_color(RED).set_fill(RED, 0.5)
        polyhedron = (
            Polyhedron(vertex_coords, faces_list)
            .set_color(RED)
            .set_fill(RED, 0.5)
            .scale(0.25)
            .rotate(PI / 2, axis=RIGHT)
            .next_to(prism, UP, buff=0)
        )
        polyhedron.set_stroke(width=0)
        camera = VGroup(prism, polyhedron).shift(5 * DOWN).scale(0.5)

        self.add(camera)
        self.add(pyramid_3d)
        polyhedron.graph.set_opacity(0)
        self.wait(1)

        self.begin_ambient_camera_rotation(rate=0.075)

        ax = (
            NumberPlane(
                x_length=40,
                y_length=40,
                x_range=(0, 40, 1),
                y_range=(0, 40, 1),
                background_line_style={
                    "stroke_color": WHITE,
                    "stroke_width": 1,
                    "stroke_opacity": 0.3,
                },
            )
            .shift(IN)
            .rotate(PI / 2, axis=OUT)
        )
        self.add(pyramid_3d)
        pyramid_left.graph.set_opacity(0)
        pyramid_right.graph.set_opacity(0)

        self.play(
            FadeIn(ax),
        )

        line = (
            Line(
                polyhedron.get_center(),
                pyramid_3d.get_corner(RIGHT + IN + DOWN) + 0.1 * (OUT + UP + LEFT),
            )
            .set_opacity(0.2)
            .set_stroke(width=3)
        )
        line2 = (
            Line(
                polyhedron.get_center(),
                pyramid_3d.get_corner(LEFT + IN + DOWN) + 0.1 * (OUT + UP + RIGHT),
            )
            .set_opacity(0.2)
            .set_stroke(width=3)
        )
        line3 = (
            Line(polyhedron.get_center(), pyramid_left.get_corner(OUT) - 0.1 * OUT)
            .set_opacity(0.2)
            .set_stroke(width=3)
        )
        line4 = (
            Line(polyhedron.get_center(), pyramid_right.get_corner(OUT) - 0.1 * OUT)
            .set_opacity(0.2)
            .set_stroke(width=3)
        )

        lines = VGroup(line, line2, line3, line4)

        self.play(Create(image_plane), FadeIn(pyramid_2d, lines))
        self.wait(2)

        camera_plane = VGroup(camera, image_plane, pyramid_2d)

        camera2 = camera_plane.copy()
        camera2.submobjects[2].shift(0.15 * RIGHT)
        camera2.rotate(PI / 15, axis=IN, about_point=ORIGIN + 10 * UP)
        camera3 = camera_plane.copy()
        camera3.submobjects[2].shift(0.15 * RIGHT)
        camera3.rotate(2 * PI / 15, axis=IN, about_point=ORIGIN + 10 * UP)
        camera4 = camera_plane.copy()
        camera4.submobjects[2].shift(0.3 * RIGHT)
        camera4.rotate(PI / 5, axis=IN, about_point=ORIGIN + 10 * UP)
        camera5 = camera_plane.copy()
        camera5.submobjects[2].shift(0.45 * RIGHT)
        camera5.rotate(4 * PI / 15, axis=IN, about_point=ORIGIN + 10 * UP)
        camera6 = camera_plane.copy()
        camera6.submobjects[2].shift(0.55 * RIGHT)
        camera6.rotate(5 * PI / 15, axis=IN, about_point=ORIGIN + 10 * UP)

        self.play(FadeIn(camera2))
        self.play(FadeIn(camera3))
        self.play(FadeIn(camera4))
        self.play(FadeIn(camera5))
        self.play(FadeIn(camera6))
        self.play(
            FadeOut(
                lines,
                ax,
            ),
        )

        self.play(
            pyramid_2d.animate.set_opacity(0.1),
            FadeOut(
                camera,
                camera2.submobjects[0],
                camera3.submobjects[0],
                camera4.submobjects[0],
                camera5.submobjects[0],
                camera6.submobjects[0],
                pyramid_3d,
            ),
        )
        pyramid_3d.shift(15 * UP + 0.2 * IN)
        self.add(pyramid_3d)
        self.stop_ambient_camera_rotation()
        self.move_camera(
            phi=90 * DEGREES,
            theta=-90 * DEGREES,
            frame_center=camera.get_center() + 18 * UP,
            run_time=2.5,
        )
        self.wait(3)
        self.play(
            pyramid_2d.animate.set_opacity(1.0),
            pyramid_3d.animate.set_opacity(0.1),
            pyramid_left.graph.animate.set_opacity(0),
            pyramid_right.graph.animate.set_opacity(0),
        )
        self.wait(3)
        self.play(
            pyramid_2d.animate.set_opacity(0.5),
            pyramid_3d.animate.set_opacity(0.25),
            pyramid_left.graph.animate.set_opacity(0),
            pyramid_right.graph.animate.set_opacity(0),
        )
        self.wait(3)


class Shifting(MovingCameraScene):
    def construct(self):
        rec = Rectangle(width=1.7, height=1.08).set_fill(config.background_color, 1.0)
        pyramid = (
            Polygon(*position_list, color=GREEN).set_fill(GREEN, 0.5).set_stroke(width=2).scale(0.2)
        ).move_to(rec.get_center())
        pyramid2 = (
            Polygon(*position_list, color=GREEN).set_fill(GREEN, 0.5).set_stroke(width=2).scale(0.2)
        ).move_to(rec.get_center() + 0.1 * LEFT)
        pyramid_2d = VGroup(rec, pyramid, pyramid2).scale(0.7)
        pyramid_l = pyramid_2d.copy().next_to(pyramid_2d, LEFT)
        pyramid_ll = pyramid_2d.copy().next_to(pyramid_l, LEFT)
        pyramid_r = pyramid_2d.copy().next_to(pyramid_2d, RIGHT)
        pyramid_rr = pyramid_2d.copy().next_to(pyramid_r, RIGHT)

        r = Rectangle(width=0.85, height=0.5, color=RED).set_fill(RED, 0.8)
        t = (
            Triangle(color=RED)
            .set_fill(RED, 0.8)
            .rotate(PI / 2)
            .scale(0.25)
            .next_to(r, RIGHT, buff=0)
        )
        camera = VGroup(r, t).scale(0.7)
        cameras = [camera.copy().shift(3 * i * RIGHT) for i in range(4)]
        camera_group = VGroup(*cameras).move_to(ORIGIN).shift(2 * DOWN)
        landmark = (
            Polygon(*position_list, color=GREEN)
            .set_fill(GREEN, 0.8)
            .set_stroke(width=2)
            .scale(0.5)
            .move_to(ORIGIN + 2 * UP)
        )
        observations = [pyramid_l, pyramid_ll, pyramid_r, pyramid_rr]
        edges = [Line(cameras[i].get_center(), landmark.get_bottom(), buff=0.5) for i in range(4)]
        for i, observation in enumerate(observations):
            observation.move_to(edges[i].get_center())

        self.add(landmark, camera_group, *edges, *observations)
        self.wait(2)
        for i, edge in enumerate(edges):
            edge.add_updater(
                lambda z, i=i: z.become(
                    Line(cameras[i].get_center(), landmark.get_bottom(), buff=0.5)
                )
            )

        base_dists = [landmark.get_center() - edges[i].get_center() for i in range(4)]
        for i, observation in enumerate(observations):
            observation.add_updater(lambda m, i=i: m.move_to(edges[i].get_center()))
            observation.submobjects[1].add_updater(
                lambda m, i=i: m.move_to(
                    edges[i].get_center()
                    + 0.1 * RIGHT * (landmark.get_center() - edges[i].get_center() - base_dists[i])
                )
            )

        self.play(landmark.animate.shift(3 * RIGHT))
        self.play(landmark.animate.shift(6 * LEFT))
        self.play(landmark.animate.shift(2 * RIGHT))
        self.wait(1)
        # Generate new random camera poses
        more_cameras = []
        for i in range(10):
            random_noise = np.random.uniform(0, 1, 3)
            more_cameras.append(
                camera.copy().move_to(
                    2 * DOWN + (random_noise * 2.5 * DOWN + (i - 4.5) * 2 * RIGHT)
                )
            )
        more_landmarks = []
        more_edges = []
        more_edge_dict = {}
        for i in range(40):
            random_noise = np.random.uniform(0, 1, 3)
            random_scale_noise = np.random.uniform(0.3, 1)
            more_landmarks.append(
                landmark.copy()
                .scale(random_scale_noise)
                .move_to(1.5 * UP + (random_noise * 2 * UP + (i - 20) * 0.5 * RIGHT))
            )
            camera_idx = min(max(int(i / 4) + int(np.random.uniform(-5, 2)), 0), 9)
            more_edge_dict[i] = camera_idx

            more_edges.append(
                Line(
                    more_cameras[camera_idx].get_center(),
                    more_landmarks[i].get_bottom(),
                    buff=0.5,
                ).set_opacity(0.5)
            )

        for edge in edges:
            edge.remove_updater(edge.updaters[0])
        self.play(
            self.camera.frame.animate.set(width=23),
            FadeIn(*more_cameras, *more_landmarks, *more_edges),
            FadeOut(*observations),
            [edge.animate.set_opacity(0.5) for edge in edges],
            run_time=2,
        )

        # Add updaters for noise
        for i, edge in enumerate(edges):
            edge.add_updater(
                lambda z, i=i: z.become(
                    Line(cameras[i].get_center(), landmark.get_bottom(), buff=0.5).set_opacity(0.5)
                )
            )
        for i, edge in enumerate(more_edges):
            edge.add_updater(
                lambda z, i=i: z.become(
                    Line(
                        more_cameras[more_edge_dict[i]].get_center(),
                        more_landmarks[i].get_bottom(),
                        buff=0.5,
                    ).set_opacity(0.5)
                )
            )

        def generate_movement(assets, noise=0.2):
            movements = []
            average_y = np.mean([asset.get_center()[1] for asset in assets])
            for asset in assets:
                random_shift = np.random.uniform(-noise, noise, 3)
                curr_pos = asset.get_center()
                curr_pos[1] = average_y
                movements.append(asset.animate.move_to(curr_pos).shift(random_shift))
            return movements

        iterations = [
            "Iteration 1, err=2.79",
            "Iteration 2, err=1.34",
            "Iteration 3, err=0.67",
            "Iteration 4, err=0.33",
            "Iteration 5, err=0.16",
            "Iteration 6, err=0.08",
            "Iteration 7, err=0.04",
            "Iteration 8, err=0.02",
            "Iteration 9, err=0.01",
        ]
        speeds = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.025, 0.025, 0.025]

        previous_iteration = None

        for i in range(8):
            iteration = Tex(iterations[i], font_size=60).move_to(7.5 * RIGHT + 5 * UP)
            if previous_iteration is not None:
                self.play(
                    AnimationGroup(
                        generate_movement(cameras + more_cameras, speeds[i]),
                        generate_movement([landmark] + more_landmarks, speeds[i]),
                    ),
                    ReplacementTransform(previous_iteration, iteration),
                )
            else:
                self.play(
                    AnimationGroup(
                        generate_movement(cameras + more_cameras, speeds[i]),
                        generate_movement([landmark] + more_landmarks, speeds[i]),
                    ),
                    FadeIn(iteration),
                )
            previous_iteration = iteration

        # self.wait(4)


class Timer(Scene):
    def construct(self):
        # Create a ValueTracker to keep track of time
        time_tracker = ValueTracker(0)

        # Create a text object that will display the timer
        timer_text = always_redraw(lambda: Tex(f"t={time_tracker.get_value():.2f} s").scale(2))
        # Add the text to the scene
        self.add(timer_text)
        # Animate the ValueTracker
        self.play(time_tracker.animate.set_value(10), run_time=10, rate_func=linear)


# To render the scene, use the comma


class Pyramid3D(ThreeDScene):
    def construct(self):
        # 3D pyramid
        vertex_coords = [[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0], [0, 0, 2.3]]
        faces_list = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 1, 2, 3]]
        pyramid = (
            Polyhedron(vertex_coords, faces_list)
            .set_color(GREEN)
            .set_fill(GREEN, 0.5)
            .set_stroke(width=0)
        )
        pyramid.graph.set_opacity(0)
        pyramid_left = pyramid.copy()
        pyramid_right = pyramid.copy().next_to(pyramid_left, RIGHT, buff=-0.15)
        pyramid_3d = (VGroup(pyramid_left, pyramid_right)).move_to(ORIGIN)

        landmark = (
            Tex("(u,v)", font_size=30)
            .rotate(PI / 2, axis=RIGHT)
            .next_to(pyramid_left.get_corner(OUT), OUT)
        )

        ax = ThreeDAxes()
        self.set_camera_orientation(phi=90 * DEGREES, theta=-90 * DEGREES)
        self.add(pyramid_3d)
        self.begin_ambient_camera_rotation(rate=-0.2)
        # self.play(FadeIn(landmark), pyramid_left.graph.animate.set_opacity(1), pyramid_right.graph.animate.set_opacity(1))
        self.wait(8)


class Formulas(Scene):
    def construct(self):
        # ref_formula = MathTex(
        #     r"\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} r_{11} & r_{12} & r_{13} & t_x \\ r_{21} & r_{22} & r_{23} & t_y \\ r_{31} & r_{32} & r_{33} & t_z \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}"
        # )
        # self.add(ref_formula)
        # pixel = MathTex(r"\(u,v\)")

        p = MathTex(
            r"\begin{bmatrix} u \\ v \\ 1 \end{bmatrix}",
        ).shift(4.5 * LEFT)
        equals = MathTex("=").shift(3.83 * LEFT)

        K = MathTex(
            r"\begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}"
        ).shift(2.1 * LEFT)
        T = MathTex(
            r"\begin{bmatrix} r_{11} & r_{12} & r_{13} & t_x \\ r_{21} & r_{22} & r_{23} & t_y \\ r_{31} & r_{32} & r_{33} & t_z \end{bmatrix}"
        ).shift(1.8 * RIGHT)
        P = MathTex(r"\begin{bmatrix} X \\ Y \\ Z \end{bmatrix}").shift(4.7 * RIGHT)
        p_brace = Brace(p, DOWN)
        p_text = MathTex(r"p_{k}").next_to(p_brace, DOWN)
        K_brace = Brace(K, DOWN)
        K_text = MathTex(r"\text{Camera Matrix } K", font_size=30).next_to(K_brace, DOWN)
        T_brace = Brace(T, DOWN)
        T_text = MathTex(r"\text{Transformation Matrix } _kT_{w}", font_size=30).next_to(
            T_brace, DOWN
        )
        P_brace = Brace(P, DOWN)
        P_text = MathTex(r"P_{w}", font_size=40).next_to(P_brace, DOWN)
        arr = VGroup(p, equals, K, T, P)
        self.play(Write(arr))
        other = VGroup(p_brace, K_brace, T_brace, P_brace, p_text, K_text, T_text, P_text)
        self.play(FadeIn(other))


class reproj(Scene):
    def construct(self):

        err_formula = MathTex(r"\text{Reprojection Error} = u_i - K T P_i")
        opti_formula = MathTex(
            r"T^* = \arg \min_T \frac{1}{2} \sum_{i=1}^n \left\| u_i - K T P_i \right\|^2"
        )

        u_i = MathTex(r"u_i - ").shift(2 * LEFT)
        br = Brace(u_i, UP)
        t = Tex(r"Measurement", font_size=30).next_to(br, UP)
        z_i = MathTex(r"KTP_i").shift(2 * RIGHT)
        br2 = Brace(z_i, DOWN)
        t2 = Tex(r"Reprojection", font_size=30).next_to(br2, DOWN)

        self.play(Write(err_formula))
        self.wait(2)
        self.play(FadeOut(err_formula))
        self.play(Write(VGroup(u_i, br, t, z_i, br2, t2)))
        self.play(FadeOut(u_i, br, t, z_i, br2, t2))
        self.play(Write(opti_formula))
        self.play(FadeOut(opti_formula))
        measurement = Tex(r"Measurement")
        reprojection = Tex(r"Reprojection")
        self.play(FadeIn(reprojection))
        self.wait(2)
        self.play(FadeOut(reprojection))
        self.play(FadeIn(measurement))
        self.wait(2)


class Tracking(Scene):
    def construct(self):
        shape_left = (
            Polygon(*position_list, color=GREEN).set_fill(GREEN, 0.8).scale(0.7).set_stroke(width=2)
        )
        shape_right = (
            Polygon(*position_list, color=GREEN)
            .set_fill(GREEN, 0.8)
            .scale(0.7)
            .set_stroke(width=2)
            .next_to(shape_left, RIGHT)
        )

        self.play(FadeIn(shape_left, shape_right))


class Disp(Scene):
    def construct(self):
        left_img_path = f"disparities/left.ppm"
        right_img_path = f"disparities/right.ppm"
        disparity_img_path = f"disparities/disp.pgm"
        left_img = np.array(Image.open(left_img_path))
        right_img = np.array(Image.open(right_img_path))
        disp_img = np.array(Image.open(disparity_img_path))

        # Crop border
        disp_img = disp_img[20:-20, 20:-20]

        left_img = ImageMobject(left_img).shift(LEFT * 3)
        right_img = ImageMobject(right_img).shift(RIGHT * 3)
        disp_img = ImageMobject(disp_img)

        left_img.height = 4
        right_img.height = 4
        disp_img.height = 4

        left_img.shift(DOWN * 6)
        right_img.shift(DOWN * 6)

        self.wait(1)
        # self.play(FadeIn(left_img, right_img))
        self.play(left_img.animate.shift(UP * 6), right_img.animate.shift(UP * 6))

        self.wait(2)

        self.play(
            AnimationGroup(
                AnimationGroup(
                    left_img.animate.shift(3 * RIGHT),
                    right_img.animate.shift(3 * LEFT).set_opacity(0.5),
                ),
                FadeIn(disp_img),
                lag_ratio=1.0,
            )
        )
        self.remove(left_img, right_img)

        high_disparity = Rectangle(width=0.5, height=0.5, fill_color=WHITE, fill_opacity=1).shift(
            2.5 * RIGHT + 0.5 * UP
        )
        low_disparity = Rectangle(width=0.5, height=0.5, fill_color=GREY, fill_opacity=1).shift(
            2.5 * RIGHT + 0.5 * DOWN
        )
        txt1 = Tex("High Disparity", font_size=30).next_to(high_disparity, RIGHT)
        txt2 = Tex("Low Disparity", font_size=30).next_to(low_disparity, RIGHT)
        self.play(
            AnimationGroup(
                disp_img.animate.shift(LEFT),
                FadeIn(high_disparity, low_disparity, txt1, txt2),
                lag_ratio=0.5,
            )
        )

        self.wait(2)


class Disp2(ZoomedScene):
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
        left_img_path = f"disparities/left.ppm"
        right_img_path = f"disparities/right.ppm"
        disparity_img_path = f"disparities/disp.pgm"
        left_img = np.array(Image.open(left_img_path))
        right_img = np.array(Image.open(right_img_path))
        disp_img = np.array(Image.open(disparity_img_path))

        # Crop border
        disp_img = disp_img[20:-20, 20:-20]

        left_img = ImageMobject(left_img).shift(LEFT * 3)
        right_img = ImageMobject(right_img).shift(RIGHT * 3)
        disp_img = ImageMobject(disp_img)

        left_img.height = 4
        right_img.height = 4
        disp_img.height = 4

        self.play(FadeIn(left_img, right_img))

        self.wait(2)

        zoomed_camera = self.zoomed_camera
        zoomed_display = self.zoomed_display
        frame = zoomed_camera.frame

        right_rect = Rectangle(width=0.75, height=0.6, color=WHITE)

        zoomed_display.move_to(2 * UP).scale(0.75)
        frame.move_to(4.9 * LEFT + 1.1 * UP)
        right_rect.move_to(4.9 * LEFT + 1.1 * UP)

        zd_rect = BackgroundRectangle(zoomed_display, fill_opacity=0, buff=MED_SMALL_BUFF)
        self.add_foreground_mobject(zd_rect)

        unfold_camera = UpdateFromFunc(zd_rect, lambda rect: rect.replace(zoomed_display))

        self.play(Create(frame), Create(right_rect))

        # self.activate_zooming()
        # self.play(right_rect.animate.shift(5 * RIGHT), self.get_zoomed_display_pop_out_animation(), unfold_camera)
        self.play(right_rect.animate.shift(5.7 * RIGHT))

        # zoomed_camera_text = Tex("Euclidean Distance", font_size=30)
        # zoomed_camera_text.next_to(zoomed_display, DOWN)
        # self.play(FadeIn(zoomed_camera_text, shift=UP))
        # self.wait()

        # self.play(
        # 	FadeOut(zoomed_camera_text),
        # )
        for i in range(4):
            self.play(
                frame.animate.shift(4.1 * RIGHT),
                right_rect.animate.shift(4.1 * RIGHT),
            )
            self.play(
                frame.animate.shift(0.3 * DOWN),
                right_rect.animate.shift(0.3 * DOWN),
            )

            self.play(
                frame.animate.shift(4.1 * LEFT),
                right_rect.animate.shift(4.1 * LEFT),
            )
            self.play(
                frame.animate.shift(0.3 * DOWN),
                right_rect.animate.shift(0.3 * DOWN),
            )

        # # Exit the zooming
        # self.play(
        #     self.get_zoomed_display_pop_out_animation(),
        #     unfold_camera,
        #     rate_func=lambda t: smooth(1 - t),
        # )
        # self.play(Uncreate(zoomed_display_frame), FadeOut(frame))
        # self.wait(2)


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
