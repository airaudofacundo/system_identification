import numpy as np
from src.model.ccx.structural_element import *

class calculix_element(structural_element):

    stiffness = np.array([])

    forward_unkno = []
    adjoint_unkno = []
    forward_hessvec_unkno = []
    adjoint_hessvec_unkno = []

    def set_element_type(self, element_type):
        self.element_type = element_type
        if element_type in ['C3D4', 'C3D6', 'C3D8', 'C3D10', 'C3D15', 'C3D20', 'CPS3', 'T3D2']:
            self.section_type = 'SOLID'

    def set_section_type(self, section_type):
        self.section_type = section_type

    def set_beam_rect(self, width, height):
        self.section_type = 'RECT'
        self.beam_width = width
        self.beam_height = height

    def set_beam_box(self, width, height, t1, t2, t3, t4):
        self.section_type = 'BOX'
        self.beam_width = width
        self.beam_height = height
        self.beam_t1 = t1
        self.beam_t2 = t2
        self.beam_t3 = t3
        self.beam_t4 = t4

    def set_beam_pipe(self, outter_radius, thickness):
        self.section_type = 'PIPE'
        self.beam_outter_radius = outter_radius
        self.beam_thickness = thickness

    def set_truss_area(self, area):
        self.truss_area = area

    def set_shell_thickness(self, thickness):
        self.section_type = 'SHELL'
        self.shell_thickness = thickness
