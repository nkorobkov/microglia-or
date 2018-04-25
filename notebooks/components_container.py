import cv2
import numpy as np
from matplotlib import pyplot as plt
import itertools
import time
import numba as nb
import copy
import logging


@nb.jit()
def change_color_of_indices(markers, indices_to_change, color):
    for i in range(markers.shape[0]):
        for j in range(markers.shape[1]):
            if markers[i, j] in indices_to_change:
                markers[i, j] = color
    return markers


@nb.jit()
def edges_close(edge1x, edge1y, edge2x, edge2y, thr):
    thrs = thr ** 2
    for i in range(len(edge1x)):
        x1 = edge1x[i]
        y1 = edge1y[i]
        for j in range(len(edge2x)):
            x2 = edge2x[j]
            y2 = edge2y[j]
            delta1 = x1 - x2
            delta2 = y1 - y2

            if (delta1 ** 2 + delta2 ** 2) < thrs:
                return True, ((x1, y1), (x2, y2))
    return False, None


class ComponentsContainer():
    def __init__(self, binary, low, verbosity=logging.DEBUG):
        self.l = logging.getLogger(self.__class__.__name__)


        self.verbosity = verbosity
        
        self.height_margin = 0
        self.width_margin = 0
        self.displayed_sq = max(binary.shape)

        nbComponents, markers, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        components_index = {}
        background_size = max(stats[:, -1])
        starttime = time.time()
        small_comp_indices = []
        for i in range(len(stats)):
            if low < stats[i, -1] < background_size:
                components_index[i] = Component(stats[i, -1], centroids[i], i)
            else:
                small_comp_indices.append(i)
        self.log_event("size prefiltering (separating big ones) took {} sec".format(time.time() - starttime))
        markers = change_color_of_indices(markers, small_comp_indices, 0)
        self.log_event("size prefiltering (deleting small ones) took {} sec".format(time.time() - starttime))

        self.binary = binary
        self.markers = markers
        self.components_index = components_index
        self.nucleus_labs = list()
        self.axon_labs = list()

        self.add_edge_info_to_components()

        self.markers_backup = np.copy(markers)
        self.components_index_backup = copy.deepcopy(components_index)

    def log_event(self, msg):
        self.l.log(self.verbosity, msg)

    def add_edge_info_to_components(self):
        starttime = time.time()
        for i in range(1, self.markers.shape[0] - 1):
            for j in range(1, self.markers.shape[1] - 1):
                cur_lab = self.markers[i, j]
                if cur_lab in self.components_index.keys() and \
                        not cur_lab == \
                                self.markers[i, j + 1] == \
                                self.markers[i, j - 1] == \
                                self.markers[i + 1, j] == \
                                self.markers[i - 1, j]:
                    self.components_index[cur_lab].add_edge(np.array([i, j]))
        self.log_event("adding edge info to components took {} sec".format(time.time() - starttime))

    @nb.jit()
    def merge_components_closer_than(self, centroid_t, contour_t):
        starttime = time.time()
        self.reload()

        pairs_to_merge = []
        for first, second in itertools.combinations(self.components_index.keys(), 2):
            close_flag, points = self.components_index.get(first).is_close(self.components_index.get(second),
                                                                           centroid_t, contour_t)
            if close_flag:
                pairs_to_merge.append((first, second, points))

        self.log_event("selecting all pairs to merge took {:3} sec".format(time.time() - starttime))
        starttime = time.time()
        for tm in pairs_to_merge:
            self._merge_two_components(*tm)
        self.log_event("merging took {:3} sec".format(time.time() - starttime))

    def reload(self):
        self.log_event("discarding previous components merge and axon-nucleus spliting")
        self.markers = copy.deepcopy(self.markers_backup)
        self.components_index = copy.deepcopy(self.components_index_backup)
        self.nucleus_labs = list()
        self.axon_labs = list()

    def split_nucl_axon(self, threshold):
        starttime = time.time()
        self.nucleus_labs = list()
        self.axon_labs = list()

        for lab in self.components_index.keys():
            if not self.components_index.get(lab).label == lab:
                continue
            if self.components_index.get(lab).size > threshold:
                self.components_index[lab] = Nucleus.from_component(self.components_index[lab])
                self.nucleus_labs.append(lab)
            else:
                self.components_index[lab] = Axon.from_component(self.components_index[lab])
                self.axon_labs.append(lab)
        self.log_event("splitting completted in {}, \naxons: {}, nucl: {}".format(time.time() - starttime,
                                                                                       len(self.axon_labs),
                                                                                       len(self.nucleus_labs)))

    def group_axons_to_nucleus(self, centroid_t, contour_t):

        starttime = time.time()
        axons_with_nucl = 0
        for nucl_lab in self.nucleus_labs:
            nucl = self.components_index.get(nucl_lab)
            for axon_lab in self.axon_labs:
                axon = self.components_index.get(axon_lab)
                close, point = nucl.is_close(axon, centroid_t, contour_t)
                if (not axon.attached) and close:
                    nucl.axons.append(axon)
                    axon.attached = True
                    axons_with_nucl += 1
        self.log_event("grouping ax to nucl took {:3} sec\n axons with nucleus: {}".format(time.time() - starttime,
                                                                                            axons_with_nucl))

    def _merge_two_components(self, survivor_label, disappearing_label, joint_points):
        disappearing_label = self.get_correct_label(disappearing_label)
        survivor_label = self.get_correct_label(survivor_label)
        if survivor_label == disappearing_label:
            return

        self.components_index[survivor_label].size += self.components_index[disappearing_label].size
        self.components_index[survivor_label].edge.extend(self.components_index[disappearing_label].edge)
        self.components_index[disappearing_label].label = self.components_index[survivor_label].label

        self.markers[self.markers == disappearing_label] = survivor_label

        cv2.line(self.markers, (joint_points[0][1], joint_points[0][0]),
                 (joint_points[1][1], joint_points[1][0]), survivor_label, lineType=4,
                 thickness=2)

    def get_correct_label(self, label):
        if label == self.components_index.get(label).label:
            return label
        else:
            return self.get_correct_label(self.components_index.get(label).label)

    def draw_components(self):

        fig = plt.figure(figsize=(11, 11), dpi=80, facecolor='w', edgecolor='k')
        plt.imshow(self.markers[self.height_margin:self.height_margin + self.displayed_sq,
                   self.width_margin:self.width_margin + self.displayed_sq],
                   cmap="jet")

    def draw_nucl_and_axons(self):
        pic = np.zeros(self.markers.shape)
        nucl_color = 400
        axon_color = 455
        color_switch = 1
        switch_val = 100
        for nucl_lab in self.nucleus_labs:
            pic[self.markers == nucl_lab] = nucl_color
            for ax in self.components_index.get(nucl_lab).axons:
                pic[self.markers == ax.label] = axon_color

            nucl_color += color_switch * switch_val
            axon_color += color_switch * switch_val
            color_switch *= 1

        fig = plt.figure(figsize=(11, 11), dpi=80, facecolor='w', edgecolor='k')
        plt.imshow(pic[self.height_margin:self.height_margin + self.displayed_sq,
                   self.width_margin:self.width_margin + self.displayed_sq],
                   cmap="jet")


class Component:
    def __init__(self, size, centroid, label):
        self.size = size
        self.centroid = centroid
        self.label = label
        self.edge = []

    @classmethod
    def from_component(cls, component):
        return cls(component.size, component.centroid, component.label, component.edge)

    def add_edge(self, point):
        self.edge.append(point)

    def is_close(self, second_comp, centroid_t, contour_t):
        if self.is_centroid_close(second_comp, centroid_t):
            return self.is_contour_close(second_comp, contour_t)
        else:
            return False, None

    def is_centroid_close(self, second_comp, thr):
        return np.linalg.norm(self.centroid - second_comp.centroid) < thr

    def is_contour_close(self, second_comp, thr):

        e1 = np.array(self.edge)
        e2 = np.array(second_comp.edge)
        return edges_close(e1[:, 0], e1[:, 1], e2[:, 0], e2[:, 1], thr)

    def is_nucleus(self):
        return False


class Nucleus(Component):
    def __str__(self):

        return "nucl size: {}, border_len: {}, axon_count: {}".format(self.size, len(self.edge), len(self.axons))

    def __init__(self, size, centroid, label, edge, axons=list()):
        super().__init__(size, centroid, label)
        if axons:
            self.axons = axons
        else:
            self.axons = list()

        self.edge = edge

    def is_nucleus(self):
        return True


class Axon(Component):
    def __str__(self):
        return "axon -- size: {}, border_len: {}".format(self.size, len(self.edge))

    def __init__(self, size, centroid, label, edge, attached=False):
        super().__init__(size, centroid, label)
        self.attached = attached
        self.edge = edge
