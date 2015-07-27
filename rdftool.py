"""Tool for calculating RDFs

"""
from __future__ import print_function

import numpy as np
from MDAnalysis.lib.distances import distance_array

from analysisbase import AnalysisBase, blocks_of


class InterRDF(AnalysisBase):
    """Analysis object for calculating intermolecular RDF.

    See the init method for arguments and keywords.

    Run the analysis with method *run*

    Results are stored in the following attributes:
     rdf
         The pair distribution function, normalised.
     edges
         The boundaries of each rdf bin.
     bins
         The center of each rdf bin.
    """
    def __init__(self, *args, **kwargs):
        """InterRDF(g1, g2, nbins=75, range=(0.0, 15.0))

        :Arguments:
          *g1*
            First AtomGroup
          *g2*
            Second AtomGroup

        :Keywords:
          *nbins*
            Number of bins in the histogram [75]
          *range*
            The size of the RDF [0.0, 15.0]
          *exclusion_block*
            A tuple representing the tile to exclude from the distance
            array. [None]
          *start*
            The frame to start at [0]
          *stop*
            The frame to end analysis at. [-1]
          *step*
            The step size through the trajectory in frames [0]

        Keyword *exclusion_block* allows same molecule contributions to
        be excluded from the rdf calculation.
        """
        self.g1 = args[0]
        self.g2 = args[1]
        self.u = self.g1.universe
        kwargs.update({'traj': self.u.trajectory})
        self._setup_frames(**kwargs)

        nbins = kwargs.pop('nbins', 75)
        hrange = kwargs.pop('range', (0.0, 15.0))

        self.rdf_settings = {'bins':nbins,
                             'range':hrange}

        # Empty histogram to store the RDF
        count, edges = np.histogram([-1], **self.rdf_settings)
        count *= 0.0
        self.count = count
        self.edges = edges
        self.bins = 0.5 * (edges[:-1] + edges[1:])

        # Need to know average volume
        self.volume = 0.0

        # Allocate a results array which we will reuse
        self._result = np.zeros((len(self.g1), len(self.g2)), dtype=np.float64)
        # If provided exclusions, create a mask of _result which
        # lets us take these out
        exclusion_block = kwargs.pop('exclusion_block', None)
        if not exclusion_block is None:
            self._exclusion_block = exclusion_block
            self._exclusion_mask = blocks_of(self._result, *exclusion_block)
            self._maxrange = hrange[1] + 1.0
        else:
            self._exclusion_block = None
            self._exclusion_mask = None

    def _singleframe(self):
        distance_array(self.g1.positions, self.g2.positions,
                       box=self.u.dimensions, result=self._result)
        # Maybe exclude same molecule distances
        if not self._exclusion_mask is None:
            self._exclusion_mask[:] = self._maxrange

        count = np.histogram(self._result, **self.rdf_settings)[0]
        self.count += count

        self.volume += self._ts.volume

    def _normalise(self):
        # Number of each selection
        nA = len(self.g1)
        nB = len(self.g2)
        N = nA * nB

        # If we had exclusions, take these into account
        if self._exclusion_block:
            xA, xB = self._exclusion_block
            nblocks = nA / xA
            N -= xA * xB * nblocks

        # Volume in each radial shell
        vol = np.power(self.edges[1:], 3) - np.power(self.edges[:-1], 3)
        vol *= 4/3.0 * np.pi

        # Number of frames
        nframes = len(self.frames)

        # Average number density
        box_vol = self.volume / nframes
        density = N / box_vol

        rdf = self.count / (density * vol * nframes)

        self.rdf = rdf

