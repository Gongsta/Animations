from manim import *


class SGMAnimation(Scene):
    def construct(self):
        # Define image dimensions
        image_width, image_height = 10, 5

        # Create grids for the left and right images
        left_image = VGroup(
            *[
                Square().scale(0.5).set_fill(BLUE_E, opacity=0.5)
                for _ in range(image_width * image_height)
            ]
        ).arrange_in_grid(rows=image_height, buff=0.1)

        right_image = (
            VGroup(
                *[
                    Square().scale(0.5).set_fill(GREEN_E, opacity=0.5)
                    for _ in range(image_width * image_height)
                ]
            )
            .arrange_in_grid(rows=image_height, buff=0.1)
            .next_to(left_image, RIGHT, buff=1)
        )

        self.add(left_image, right_image)

        # Animate comparison of corresponding pixels
        for i in range(image_width * image_height):
            self.play(
                left_image[i].animate.set_fill(RED, opacity=0.8),
                right_image[i].animate.set_fill(RED, opacity=0.8),
                run_time=0.1,
            )
            self.wait(0.1)
            self.play(
                left_image[i].animate.set_fill(BLUE_E, opacity=0.5),
                right_image[i].animate.set_fill(GREEN_E, opacity=0.5),
                run_time=0.1,
            )

        # Display text indicating cost calculation
        cost_text = Text("Calculating Costs...").next_to(left_image, DOWN, buff=1)
        self.play(Write(cost_text))
        self.wait(2)
        self.play(FadeOut(cost_text))
