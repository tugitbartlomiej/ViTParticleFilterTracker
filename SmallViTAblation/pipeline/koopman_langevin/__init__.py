"""Koopman and extended-Langevin analysis of image latents for frame selection.

Everything here is label-free at scoring time. Ground-truth boxes, tips and
velocities from Cataract-LMM are read only by the validation functions, never
by the functions that produce saliency, tool state, scores or selections.
"""
