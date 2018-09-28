# CLAM-SPOPT

We provide the source code implementing the meat of our approach[^1]. The
code in its current is raw, still containing research code and unused methods.
We experiment scripts and plotting scripts will be released shortly.

[^1]: Clement Gehring, Tomas Lozano-Perez, Leslie Pack Kaelbling. Adaptable replanning with compressed linear action models for learning from demonstrations. *Proceedings of the Conference on Robot Learning (CoRL)*. 2018.

## CONTENTS:
* python/episodelistener.py :
		provides the class definition needed to load our data. For final release,
		the data will be converted in a implementation independent format.

* python/interfaces.py :
		defines some basic interfaces used elsewhere in the code.

* python/mathtools.py :
		provides the implementation for the incremental low-rank matrix compression
		as well as a few other useful functions used elsewhere.

* python/model.py :
		provides the implementation for the CLAMs under the class LowRankModel

* python/NavCar.py : 
		provides implementations for the various part of the race track domain
		with LIDAR

* python/planning.py :
		provides implementation of AVI with CLAMs in class LowRankAVI, of SPOPT
		with CLAMs in LowRankGradientPlanner

* tracks/ :
		pickled instances of every track presented in the paper, note that the
		track numbering was changed in the paper in order to group the harder tracks
		together, e.g., 4 and 5 from the paper. This will be properly labelled
		when publicly released. Additionally, tracks will be converted to an
		implementation independent format.

* data/ :
		pickled instances of an episodelistener containing the training data for
		each track. Follows the same numbering as the tracks directory. This will 
		be numbered consistently with respect to the paper when publicly released.
		Additionally, the trajectories will be converted to an implementation
		independent format.

