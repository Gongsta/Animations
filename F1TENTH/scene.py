from manim import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import distance_transform_edt
from skimage import morphology as morph
from skimage.morphology import skeletonize

config.background_color = "#1f1f1f"

class OccupancyGrid(Scene):
	def construct(self):
		size = 10
		squares = [Square(side_length=0.5) for _ in range(size**2)]
		squares_group = VGroup(*squares)
		squares_group.arrange_in_grid(rows=size, cols=size, buff=0)
		self.play(Create(squares_group))
		self.wait(2)
		# self.play(squares[0].animate.set_fill(RED, opacity=0.5)) # Not ideal because other squares are drawn on top of it
		# new_square = Square(side_length=0.5, color=WHITE, fill_opacity=0.5).move_to(squares[1].get_center())
		# self.play(Create(new_square))
		self.wait(2)

class AngleScene(Scene):
	def construct(self):
		lidar = ImageMobject("car.png").scale(0.15).rotate(90 * DEGREES)

class LidarScanAnimation(Scene):
	def construct(self):
		# Create a grid
		grid_size = 20
		grid_group = VGroup()
		grid = []
		for i in range(grid_size):
			row = []
			for j in range(grid_size):
				cell = Square(side_length=0.25).set_stroke(width=0.2)
				cell.move_to(i*0.25*RIGHT + 1*LEFT + j*0.25*UP + 1 * DOWN)
				cell.set_fill(WHITE, opacity=0)
				row.append(cell)
				grid_group.add(cell)
			grid.append(row)

		grid_group.center()

		# Place the lidar scanner at the center of the grid
		

		lidar = ImageMobject("car.png").scale(0.15).rotate(90 * DEGREES).move_to(grid_group.get_center())
		self.play(GrowFromCenter(lidar))

		# Define lidar scan properties
		num_scans = 30
		max_dist = grid_size / 2

		# Simulate lidar scans - fake distance measurements at specific angles
		angles = np.linspace(- np.pi/4, np.pi + np.pi/4, num_scans)
		np.random.seed(6)
		distances = np.random.uniform(low=max_dist/2, high = max_dist, size=(num_scans))
		
		line = Line(lidar.get_center(), lidar.get_center(), color=RED)
		self.add(line)

		# Process each lidar scan individually
		calculations = None
		count = 0
		for theta, r in zip(angles, distances):
			x = int((r * np.cos(theta)) + grid_size / 2)
			y = int((r * np.sin(theta)) + grid_size / 2)

			# Update the calculations text
			if count < 7:
				new_calculations = MathTex(
									"\\text{Distance } r &= ", f"{r:.1f} \\space m", 
									"\\text{  Angle } \\theta = ", f"{np.degrees(theta):.1f}^\\circ",
									tex_environment="align*",
									font_size=40, 
									)
				new_calculations.shift(2.5 * UP)

			else:
				new_calculations = MathTex(
									"&\\underline{\\text{Original Measurement}}", "\\\\",
									"r &= ", f"{r:.1f} \\space m", 
									"\\text{   } \\theta = ", f"{np.degrees(theta):.1f}^\\circ", "\\\\",
									"\\\\",
									"&\\underline{\\text{Projection on Occupancy Grid}}", "\\\\",
									f"x &= {r:.1f}m\\cos({np.degrees(theta):.1f}^\\circ )", f"= {(r * np.cos(theta)):.1f} m",  "\\\\",
									f"y &= {r:.1f}m\\sin({np.degrees(theta):.1f}^\\circ )", f"={(r * np.sin(theta)):.1f} m", "\\\\",
									tex_environment="align*",
									font_size=30, 
									)
				new_calculations.move_to(RIGHT * 3)


			if calculations is None:
				calculations = new_calculations

			if count >= 7:
				cell = grid[x][y]
				end_point = cell.get_center()
				new_line = Line(lidar.get_center(), end_point, color=RED)
				if count == 7:
					new_line.shift(3*LEFT)
					grid_group.shift(3*LEFT)
					self.play(FadeOut(calculations), lidar.animate.shift(3*LEFT).scale(0.35), line.animate.shift(3*LEFT), 
	       					FadeIn(grid_group), Transform(line, new_line), cell.animate.set_fill(WHITE, opacity=0.6), FadeIn(new_calculations), run_time=0.5)
				else:
					self.play(Transform(line, new_line), cell.animate.set_fill(WHITE, opacity=0.6), TransformMatchingTex(calculations, new_calculations), run_time=0.5)
				self.wait(0.1)

			else:
				cell = grid[x][y]
				end_point = cell.get_center()
				new_line = Line(lidar.get_center(), end_point, color=RED)


				if count < 4:
					self.play(Transform(line, new_line), run_time=0.5)
				elif count == 4:
					self.play(Transform(line, new_line), FadeIn(new_calculations), run_time=0.5)
				else:
					self.play(Transform(line, new_line), TransformMatchingTex(calculations, new_calculations), run_time=0.5)
				cell.set_fill(WHITE, opacity=0.6)
				self.wait(0.2)

			count += 1
			calculations = new_calculations



		self.wait(2)
