"""Base functions for RDF creating tools

"""

from __future__ import print_function

import numpy as np
import logging
#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AnalysisBase(object):
    """Base class for defining multi frame analysis

    Defines common functions for setting up frames to analyse
    """
    def _setup_frames(self, **kwargs):
        """Look for the keywords which control trajectory iteration
        and build the list of frames to analyse.

        Returns an array of the identified frames
        """
        self.trajectory = kwargs.pop('traj', None)
        if self.trajectory is None:
            raise ValueError("Must supply the 'traj' keyword")

        nframes = len(self.trajectory)

        start = kwargs.pop('start', 0)
        stop = kwargs.pop('stop', nframes)
        skip = kwargs.pop('skip', 1)

        logger.debug("_setup_frames")
        logger.debug(" * settings: start {} stop {} skip {}".format(
            start, stop, skip))

        frames = np.arange(start, stop, skip)

        logger.debug(" * identified frames:\n"
                     " * {}".format(frames))
        self.frames = frames

    def _setup(self):
        """Set up data structures for your results

        Called first thing inside run().
        """
        pass

    def _singleframe(self):
        """Calculate data from a single frame of trajectory
 
        Don't worry about normalising, just deal with a single frame.
        """
        pass

    def _normalise(self):
        """Finalise the results you've gathered.

        Called at the end of the run() method to finish everything up.
        """
        pass

    def run(self):
        for frame in self.frames:
            logger.info("Seeking frame {}".format(frame))
            self._ts = self.trajectory[frame]
            print("Doing frame {} of {}".format(frame, self.frames[-1]))
            logger.info("--> Doing single frame")
            self._singleframe()
        logger.info("Applying normalisation")
        self._normalise()

    def __iter__(self):
        def iter_frames():
            self._setup()
            for i in self.frames:
                self._ts = self.trajectory[i]
            yield self._singleframe()

        return iter_frames()


def blocks_of(a, n, m):
    """Extract a view of (n, m) blocks along the diagonal.
 
    Arguments:
      a - starting array
      n, m - size of each miniblock
 
    Returns:
      (nblocks, n, m) view of the original array. 
      Where nblocks is the number of times the miniblock fits in the original.
 
    n, m must divide a into an identical integer number of blocks.
 
    based on: 
    http://stackoverflow.com/a/10862636
    but generalised to handle non square blocks.
 
    Uses strides so probably requires that the array is C contiguous
 
    Returns a view, so editing this modifies the original array
    """
    nblocks = a.shape[0] / n
 
    if not nblocks == a.shape[1] / m:
        # Can't have any remained when dividing into chunks in each direction
        # Must also create the same number of chunks in each direction
        raise ValueError("Must divide into same number of blocks in both directions")
 
    new_shape = (nblocks, n, m)
    new_strides = (n * a.strides[0] + m * a.strides[1],
                   a.strides[0], a.strides[1])
 
    return np.lib.stride_tricks.as_strided(a, new_shape, new_strides)

        
