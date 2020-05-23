import sys
import argparse
import logging
import pickle
import re
import copy
import shutil
from os import listdir, walk, path
from hashlib import sha256
from operator import itemgetter
from sys import argv
from collections import namedtuple, defaultdict
from enum import Enum
from time import time
from itertools import groupby

# 'local' modules
import lazy_groupby, filetree, compute_moves, misc

"""
A python 3 script for path mirrorring, that attempts to minimize file transfers
between reference and destination dir. It is useful for instance to synchronize a
local dir with a network share.

There is NO WARRANTY WHATSOEVER regarding the correct behavior of this script.
Use with caution, and review the changes proposed before accepting them (the
script gives the user the opportunity to do that).

The file transfers are minimized by ensuring that all the files in source that
are available on the destination side are copied or moved inside destination
instead of being transferred from source.

If some files were renamed or reoragnized in source after a previous
synchronization for instance, this enables to propagate these renames without
any file copy from source to destination.

The scripts uses a user-selectable logic to determine whether files are equal.
the default logic is loose and only relies on file size and modification time.
It may therefore produce false-positives (files wrongly assumed to be equal)
and false-negatives. Use of stricter logic ruling out false-positives is possible,
('medium' and 'strict' settings), but negates most of the benefits of this tool,
since these stricter settings require accessing the contents of possibly remote
files.

False-positives, false-negatives or failure to complete operations may lead to
an incoherent state with files being deleted on the destination side, but in
such a case the source side is still available to change the comparison logic
or retry the operation.

False positives may lead to a seemingly complete transfer where destination
is in	fact different from source. False negatives on the contrary lead to a
correct	end-state, but with unneeded file transfers.

There is no provision for the management of symbolic links.

Usage; file comparison heuristics
---------------------------------

basic usage is as follows:

> mirror.py compare <source dir> <destination dir>

policy is selected with --policy <pol>, <pol> = strict, medium or loose.

loose:
files are considered equal if they have the same size and modification time
(mt). When there are more than two files with the same size and mt, if they
form pairs with the same name, with for each pair one file in source and one
in destination, files inside each pair are considered equal and files in
different pairs are considered to be different.

medium:
files are considered equal if they have the same size, modification time
and partial hash (cryptographic hash of three extracts at the beginning,
middle and end of the file). File names are not considered at all.

strict:
files are considered equal if they have the same size, modification time
and full hash. partial hash is still computed before full hash to discriminate
files. File names are not considered at all.

Comparison criteria are applied sequentially to discriminate files: mt will
not be used to compare files of different sizes for instance. Therefore
hashes are computed only when necessary.

Option -mt_tol <num> adds a tolerance, in seconds, to modification times,
everywhere modification times are used. Files are in the same class
w.r.t. modification times if their modification times differ by less than
the tolerance (*).

the result of a comparison can be dumped to a file with option --dump <file>:

> mirror.py compare --dump <file> <source dir> <destination dir>   (A)

The operations to perform the synchronization can be computed from that file
as follows:

> mirror.py reload <file>                                          (B)

This is useful essentially for debugging purposes. For the computed operations
to make sense, the two executions of the script (A) and (B) should be performed
in the same directory and with <source dir> and <destination dir> unchanged.

(*) there is a more complex logic at work when more than 2 files needs to be
discriminated. The mt criterion then becomes more lax. See code.

--logging accepts standard python logging arguments. See
https://docs.python.org/3/library/logging.html#levels.

Steps of the algorithm
----------------------

- Files are listed on source and destination side and are put into equivalence
  classes based on the logic chosen. Equivalent files are deemed equal.

- Subirectories from reference and destination are matched with a heuristic
  based on the number of common and different files (including already matched
  subdirectories). This enables to detect directory renames or moves and to
  avoid doing full dir deletes and copies in such cases.

- Destination subdirectories are shuffled to mimic the structure of source
  directories according to the directory matching.

- Remaining file differences are resolved through file moves/copies/deletes,
  using destination files whenever possible.

After the operations needed to make destination equal to source are computed,
they are tested on a virtual representation of the destination file tree. If the
result is equal to the source file tree (where file equality is decided according
to the logic chosen by the user), the user is presented with the operation list
and decides whether to apply it.

Known issues
------------
1)	Paths with spaces are supported but must be enclosed at the python level with
	"". Since the shell usually removes these chars but needs them itself for space
	handling, this means that the string must be enclosed at the shell level with
	"\" .... \""
2)	This script assumes that Unicode strings are writable on the console.
	This is not true out-of-the-box on Windows before python 3.6, and  can be
	fixed with an install of win-unicode-console, which  can be configured to be
	enabled system-wide. See https://github.com/Drekin/win-unicode-console
	and
	https://docs.python.org/3/tutorial/appendix.html#the-customization-modules
	which tells you how you can add the win-unicode-console initialization code
	to your python install.
3)	Under Windows, redirection of the script output to a file does not work out of
	the box (even with python >= 3.6). To fix that, define
	PYTHONIOENCODING=utf-8
	before running the script.
4)	Performance for transferring large files on Windows is only optimal
	starting with python 3.8.
"""

file_info = namedtuple("file_info", ["top_dir_idx", "relpath_tuple", "size", "mt"])

"""
An enum for the choice of policy to test whether files are different.
"""
class split_policy(Enum):
	looser = -1
	loose = 0
	medium = 1
	strict = 2

if sys.platform == "win32":
	def norm_path(p):
		"""
		Normalizes a path to the "UNC" syntax to handle long file paths
		Used to handle long paths in actual file operations.
		NOT used for internal	representation of file paths.
		"""
		uncprefix = "\\\\?\\UNC"
		if p[:2] == uncprefix[:2] and p[:7] != uncprefix:
			fp = uncprefix + p[1:]
		else:
			fp = "\\\\?\\" + p
		return fp
else:
	def norm_path(p):
		return p

def filedata(dirname, filename):
	"""
	returns a 4-uple (dirname, filename, file size, file modification time)
	for file 'filename' in directory 'dirname'
	"""
	filepath = norm_path(path.join(dirname, filename))
	sz = 0
	try:
		sz = path.getsize(filepath)
		mt = path.getmtime(filepath)
	except FileNotFoundError:
		print("file " + filename + " not found")
		raise
	return sz, int(mt)


def partial_digest(full_file_path, file_size):
	"""
	Compute a digest of file with path 'full_file_path', and of size 'size'.
	This digest is a sha-256 hash of the beginning, middle and end 1KB blocks,
	using the full file if the file size is below 3 x 1KB.
	"""
	npath = norm_path(full_file_path)
	with open(npath, "rb") as f:
		if file_size > 3072:
			data = f.read(1024)
			f.seek(1024 + (file_size - 3072) // 2, 0)
			data += f.read(1024)
			f.seek(-1024, 2)
			data += f.read(1024)
		else:
			data = f.read(file_size)
	return sha256(data).hexdigest()


def full_digest(full_file_path):
	"""
	Compute a digest of file with path 'full_file_path'.
	This digest is a sha256 hash of the file.
	"""
	npath = norm_path(full_file_path)
	hasher = sha256()
	with open(npath, "rb") as f:
		data = f.read(1024)
		while len(data) > 0:
			hasher.update(data)
			data = f.read(1024)
		return hasher.hexdigest()


def split(elements_with_keys):
	# elements_with_keys = [(e_1, k_1), ... (e_n,k_1),(e_n1+1,k_2),...,(e_n2,k2),(e_n2+1,k3),...]
	# group by same key
	new_groups = groupby(sorted(elements_with_keys, key=itemgetter(1)), itemgetter(1))
	class_split = [[e_with_key[0] for e_with_key in g] for k, g in new_groups]
	return class_split


def split_with_tol(elements_with_keys, tol):
	if elements_with_keys == []:
		return []
	elements_sorted = sorted(elements_with_keys, key=itemgetter(1))
	result = []
	current_class = []
	current_key = elements_sorted[0][1]
	for e, k in elements_sorted:
		if k <= current_key + tol:
			current_class.append(e)
		else:
			result.append(current_class)
			current_class = [e]
		current_key = k
	result.append(current_class)
	#print('split_with_tol result : {} (tol={})'.format(result, tol))
	return result

# for this splitting function, the keys are considered to be hints that elements are equal, i.e. they are
# not used to separate elements when they are different, but to pair matching elements.
# in other words, when elements in source and destination can all be paired with the help of the attribute,
# they are split in pairs. otherwise, the class is left as-is, leaving the work to more expensive attributes
# (typically digests)
def split_if_pairs(elements_with_keys, list_files):
	# elements_with_keys = [(e_1, k_1), ... (e_n,k_1),(e_n1+1,k_2),...,(e_n2,k2),(e_n2+1,k3),...]
	elts = [list(map(lambda p:p[0], elements_with_keys))] #default answer if we are not sure
	e0 = list(filter(lambda t: list_files[t[0]].top_dir_idx == 0, elements_with_keys))
	e1 = list(filter(lambda t: list_files[t[0]].top_dir_idx == 1, elements_with_keys))
	if len(e0) != len(e1):
		return elts

	def build_dict(elt_list):
		d = {}
		for e,n in elt_list:
			if n in d:  # one file allowed per name
				return {} # answer meaning failure
			else:
				d[n] = e
		return d

	d0 = build_dict(e0)
	d1 = build_dict(e1)

	if(d0 == {}) or (d1=={}):
		return elts

	result = []
	for n in d0:
		if not n in d1:
			return elts
		else:
			result.append([d0[n], d1[n]])
	return result


def traverse_trees(source, destination, policy, mt_tol):
	"""
	source and input dirs are assumed to be normalized through
	os.path.normath.
	- all paths stored are relative to their top dir, in list form
	(list of dir or file names), also called 'relpath'
	- top_dir_idx is 0 for source, 1 for destination.

	Produces a dictionary with the following fields:
	'source', 'destination':
		a copy of the input parameters
	'list_files' :
		a list of files found in source or in destination.
			each file is represented by a file_info named tuple with the following data:
			top_dir_idx (see above), relpath_tuple (path including file name in tuple form),
			size, mtime (last modification time)
	'list_dirs':
		a list of lists. list_dirs[top dir idx] is the list of subdirectories of top dir.
		'list_leaves':
			same as above, but only for subdirs that do not have sudirectories themselves.
	'files':
		a list of maps. files[top dir idx][relpath] is the list of
			(non-directory) files found in subdir 'relpath' of top dir.
	'list_classified_files':
		list of lists of files assumed to be identical
			(according to the heuristic described below).
			Each file is represented by the index in list_files of its
		file_info object.
	'filesystems':
			filesystems[top dir idx] is a tree representation of the the files found in top dir.

	file comparison heuristics:
	3 heuristics can be chosen through the 'policy' argument. See enum definition.
	A lazy mechanism ensures that partial and full digests are only computed
	when needed.
	"""
	path_list = [source, destination]

	def full_path(f_info):
		return path.join(path_list[f_info.top_dir_idx], *f_info.relpath_tuple)

	class OutsideTreeError(Exception):
		pass

	def abs_to_relpath(i, abs_path):
		if not abs_path.startswith(path_list[i]):
			raise OutsideTreeError()
		return abs_path.replace(path_list[i], "", 1).lstrip(path.sep)

	def strict_class_is_final(properties_set, class_file_list):
		# separate files by all available criteria; no early abort except for length-1 classes.
		return len(class_file_list) == 1

	def loose_class_is_final(properties_set, class_file_list):
		if len(class_file_list) == 1:
			return True
		if not ("size" in properties_set and "mt" in properties_set):
			return False
		# from here on, files have same size approximate same mt.
		#print('loose_class_is_final input: {}'.format(class_file_list))
		if (len(class_file_list) == 2) and set(list_files[i].top_dir_idx for i in class_file_list) == set((0, 1)):
			return True
		if "partial_digest" in properties_set:
			return True
		return False

	def f_attr_partial_digest(i):
		return partial_digest(full_path(list_files[i]), list_files[i].size)

	def f_attr_full_digest(i):
		return full_digest(full_path(list_files[i]))

	def f_attr_size(i):
		return list_files[i].size

	def f_attr_name(i):
		return list_files[i].relpath_tuple[-1]

	def f_attr_mt(i):
		return list_files[i].mt

	def f_split_mt(c):
		return split_with_tol(c, mt_tol)

	def f_split_if_pairs(c):
		return split_if_pairs(c, list_files)

	list_files = []
	list_dirs = [[], []]
	list_leaves = [[], []]
	files = [{}, {}]

	#build the representation of the source and target dirs
	fs = [filetree.filesystem(), filetree.filesystem()]
	for i, p in enumerate(path_list):
		print("traversing tree {0}".format(i))
		for dirname, dirnames, filenames in walk(p):
			# 'dirname' walks through all subdirs of p. (absolute path)
			# 'dirnames' is the list of dirs in 'dirname' (relative paths)
			#  'filenames' is the list of all non-dir files in 'dirname'.
			relpath = filetree.filesystem.split_path(abs_to_relpath(i, dirname))
			if len(relpath) > 0:  # don't add root dir
				fs[i].create(relpath[:-1], relpath[-1], is_dir=True)
			for f in filenames:
				fs[i].create(relpath, f, is_dir=False)
			list_dirs[i].append(relpath)
			if len(dirnames) == 0:
				list_leaves[i].append(relpath)
			files[i][relpath] = filenames
			list_files.extend(
				[
					file_info(i, relpath + (filename,), *filedata(dirname, filename))
					for filename in filenames
				]
			)

	if policy == split_policy.loose:
		filters = [
			("size", f_attr_size, split),
			("mt", f_attr_mt, f_split_mt),
			("name", f_attr_name, f_split_if_pairs),
			("partial_digest", f_attr_partial_digest, split),
		]
		class_is_final = loose_class_is_final
	elif policy == split_policy.medium:
		filters = [
			("size", f_attr_size, split),
			("mt", f_attr_mt, f_split_mt),
			("partial_digest", f_attr_partial_digest, split),
		]
		class_is_final = strict_class_is_final
	else:
		filters = [
			("size", f_attr_size, split),
			("mt", f_attr_mt, f_split_mt),
			("partial_digest", f_attr_partial_digest, split),
			("full_digest", f_attr_full_digest, split),
		]
		class_is_final = strict_class_is_final

	classes = lazy_groupby.lazy_groupby(
		list(range(len(list_files))), filters, class_is_final
	)

	#to print the final classes
	#print([list_files[i] for c in classes for i in c])

	ok = fs[0].validate() and fs[1].validate()
	result = {
		"source": source,
		"destination": destination,
		"list_files": list_files,
		"list_dirs": list_dirs,
		"list_leaves": list_leaves,
		"files": files,
		"list_classified_files": classes,
		"filesystems": fs,
	}
	return ok, result


def analyze(pair_data):
	destination = pair_data["destination"]
	list_files = pair_data["list_files"]
	list_dirs = pair_data["list_dirs"]
	list_leaves = pair_data["list_leaves"]
	files = pair_data["files"]
	list_classified_files = pair_data["list_classified_files"]
	fs = pair_data["filesystems"]

	# associate to each file a file identifier (an int), such that file ids are
	# equal iff files are identical.
	# file_idx_map[(top dir idx, relative path, file)] = file_id
	# file_idx_map_back[file_id] = list of files with this id. list elt format:
	# (top dir idx, relative path, filename)
	# dir content [top dir idx][relpath] = set of file indexes of files in top dir/relpath
	# (only at root level)
	file_idx_map = {}
	# reverse map
	# we copy list_classified_files
	file_idx_map_back = list(list_classified_files)
	# because we will add elements to the copy and we don't want these changes to
	# last.

	for class_idx, file_class in enumerate(list_classified_files):
		for i in file_class:
			f = list_files[i]
			file_idx_map[(f.top_dir_idx, f.relpath_tuple)] = class_idx
			fs[f.top_dir_idx].add_metadata(f.relpath_tuple, "id", class_idx)
	dir_contents = [{}, {}]
	for top_dir_idx, l in enumerate(list_dirs):
		for relpath in l:
			dir_contents[top_dir_idx][relpath] = set(
				[
					file_idx_map[(top_dir_idx, relpath + (f,))]
					for f in files[top_dir_idx][relpath]
				]
			)

	# associates subdirs of source and destination, starting with leaves,
	# attempting (heuristically) to reduce the number of additions / deletions
	# needed to make each associated dir identical to its counterpart

	matches_stats = namedtuple("matches_stats", ["intersection", "penalty"])

	dir_association = {}
	associated_dirs = set()
	processed_ref_dirs = set()
	current_leaves = set(list_leaves[0])
	next_leaves = set()
	file_matches_count = defaultdict(int)
	files_count = defaultdict(int)

	logger = logging.getLogger(__name__)

	def stat_w_lowest_penalty(d):
		default_stat = matches_stats(intersection=0, penalty=len(dir_contents[0][d]))
		return min(
			matches[d].values(), key=lambda stats: stats.penalty, default=default_stat
		)

	while len(current_leaves) > 0:
		matches = {}
		for d in current_leaves:
			if len(d) == 0:
				# don't try to associate the root of reference to dir to a subdir,
				# since no subdir can be moved to root
				continue
			logger.info("analyze(): processing dir {}".format(d))
			# build the list of file class indexes found in d
			file_class_indexes = dir_contents[0][d]
			dir_set = set()
			files_count[d] += len(file_class_indexes)
			for class_idx in file_class_indexes:
				for i in file_idx_map_back[class_idx]:
					f = list_files[i]
					if f.top_dir_idx == 1:
						dir_set.add(f.relpath_tuple[:-1])
						file_matches_count[d] += 1
			# dir_set is the set of dirs in target that contain at root level
			# at least one file found at root level in d.
			data = {}
			for dprime in dir_set:
				file_indexes_prime = dir_contents[1][dprime]
				intersec_sz = len(file_indexes_prime.intersection(file_class_indexes))
				assert intersec_sz > 0
				# penalty_sym_diff = # of symmetric difference between file sets
				penalty_sym_diff = (
					len(file_indexes_prime)
					+ len(file_class_indexes)
					- 2 * intersec_sz
				)
				if d == dprime:
					penalty_name = -2
				elif len(dprime) > 0 and d[-1] == dprime[-1]:
					penalty_name = -1
				else:
					penalty_name = 0
				data[dprime] = matches_stats(
					intersection = intersec_sz,
					penalty = 3 * penalty_sym_diff + penalty_name,
				)
			matches[d] = data
		# matches[d][dprime] = [intersect(d, d'), penalty(d,d')]
		#  for all d' s.t. intersect(d,d') > 0.
		# dirs_with_interect and dirs_sorted are built to compute a 'good' processing
		# order for dirs d, where high-quality matches are selected first.
		# dirs_with_intersect[i] = [(d, [intersect(d, d'), penalty(d, d')])]
		# where for each d, d' is a dir with minimum penalty(d, d').
		# dirs_sorted is the list of d's appearing in dirs_with_intersect, sorted by decreasing intersect_d'.
		dirs_with_intersect = [(d, stat_w_lowest_penalty(d)) for d in matches]
		dirs_sorted = [
			x[0]
			for x in sorted(
				dirs_with_intersect, key=lambda x: x[1].intersection, reverse=True
			)
		]
		# we process pairs of dirs that have a large intersection first. This enables us to 'lock'
		# the association of these dirs that are more fruitful than others.
		logger.info("ref dir <- (built from) destination dir:")
		for d in dirs_sorted:
			# d may already have been encountered and associated from a shorter path from some leave,
			# in that case do not overwrite association
			if d in dir_association:
				continue
			# candidates is a sub-dict of matches[d]
			# where already associated dirs and root are excluded from candidate matches
			# (destination root cannot be moved into another dir)
			candidates = {
				dprime: stats
				for dprime, stats in matches[d].items()
				if not dprime in associated_dirs and len(dprime) > 0
			}
			processed_ref_dirs.add(d)
			if len(candidates) > 0:
				# choose candidate with minimum penalty
				# FIXME we are doing again the minimum search done when computing dirs_with_intersect
				dprime = min(candidates, key=lambda d: candidates[d].penalty)
				dir_association[d] = dprime
				associated_dirs.add(dprime)
				if d != dprime:
					logger.info(
						" {} <- {}".format(path.sep.join(d), path.sep.join(dprime))
					)
				if len(d) > 0 or len(dprime) > 0:
					curr_idx = len(file_idx_map_back)
					file_idx_map_back.append([])
					if len(d) > 0:
						p = d[:-1]
						next_leaves.add(p)
						dir_contents[0][p].add(curr_idx)
						file_matches_count[p] += file_matches_count[d]
						files_count[p] += files_count[d]
						dir_descriptor = file_info(0, d, 0, 0)
						file_idx_map_back[-1].append(len(list_files))
						list_files.append(dir_descriptor)
					if len(dprime) > 0:
						pprime = dprime[:-1]
						dir_contents[1][pprime].add(curr_idx)
						dir_descriptor = file_info(1, dprime, 0, 0)
						file_idx_map_back[-1].append(len(list_files))
						list_files.append(dir_descriptor)
			else:  # len(candidates) == 0
				# there is no dir with at least one file in common with d in destination
				dir_association[d] = ""
				if len(d) > 0:
					curr_idx = len(file_idx_map_back)
					file_idx_map_back.append([])
					p = d[:-1]
					next_leaves.add(p)
					file_matches_count[p] += file_matches_count[d]
					files_count[p] += files_count[d]
					dir_contents[0][p].add(curr_idx)
					dir_descriptor = file_info(0, d, 0, 0)
					file_idx_map_back[curr_idx].append(len(list_files))
					list_files.append(dir_descriptor)
		current_leaves = next_leaves
		next_leaves = set()

	# check that all file indexes are valid.
	for c in list_classified_files:
		for i in c:
			assert i < len(list_files)

	# dirs in reference with no file in destination (including in subdirs),
	# and which are maximal for this property (i.e. paths are shortest).
	# root is forbidden because we can' t copy the root onto the destination root.
	dirs_with_no_matches = set(
		d
		for d in file_matches_count
		if file_matches_count[d] == 0
		and (len(d) > 0 and (len(d) == 1 or file_matches_count[d[:-1]] > 0))
	)
	dirs_with_files = set(d for d in files_count if files_count[d] > 0)
	# these dirs could be copied from reference, instead of copying
	# files and subdirs one by one.
	logger.info(
		"Dirs with at least one file and with no common content with destination:"
	)
	dirs_to_copy = dirs_with_no_matches.intersection(dirs_with_files)
	for f in sorted(dirs_to_copy):
		logger.info(f)

	ref_sources = set(list_dirs[0])
	ref_sources.remove(())
	# ensure all subdirs were considered for association
	assert ref_sources == processed_ref_dirs

	# ensure dir move map can be reversed and build the reverse map
	dir_association_reverse = {}
	for d in dir_association:
		assert not dir_association[d] in dir_association_reverse
		dprime = dir_association[d]
		if len(dprime) > 0:
			dir_association_reverse[dir_association[d]] = d
	if not destination in dir_association_reverse:
		dir_association_reverse[destination] = destination
	dir_moves = {}
	dir_moves["file_idx_map"] = file_idx_map
	dir_moves["dir_association"] = dir_association
	dir_moves["dir_association_reverse"] = dir_association_reverse
	dir_moves["dirs_to_copy"] = dirs_to_copy
	return dir_moves



def compute_modifications(pair_data, dir_moves):
	display_time = False
	tstart = t0 = t1 = time()
	destination = pair_data["destination"]
	list_files = pair_data["list_files"]
	list_dirs = pair_data["list_dirs"]
	files = pair_data["files"]
	list_classified_files = pair_data["list_classified_files"]
	filesystems = pair_data["filesystems"]

	dir_association = dir_moves["dir_association"]
	dir_association_reverse = dir_moves["dir_association_reverse"]
	dirs_to_copy = dir_moves["dirs_to_copy"]
	# keep only files in both dirs

	if display_time:
		t1 = time()
		print("A: time: {:.3f}".format(t1 - t0))
		t0 = t1

	# Cache the path-file --> node id map before doing any change to the destination
	# file system representation.
	# This enables us to find the id of a file from its initial path. The filesystem can
	# then provide the file path of the file corresponding to a node id at any time,
	# after various move operations.
	nid_dict = {
		list_files[i]
		.relpath_tuple: filesystems[1]
		.path_to_node_id(list_files[i].relpath_tuple)
		for file_class in list_classified_files
		for i in file_class
		if list_files[i].top_dir_idx == 1
	}

	# compute the current path of a file given its initial path.
	# relies on the fact that node ids of deleted files are not reused.
	def current_path(relpath):
		return filesystems[1].node_path(nid_dict[relpath])

	# computes the node id of a file given by its path in the target fs.
	def dest_idx(relpath):
		return filesystems[1].path_to_node_id(relpath)

	common_files = [
		file_class
		for file_class in list_classified_files
		if {list_files[i].top_dir_idx for i in file_class} == {0, 1}
	]
	# files only in 0
	files_0 = [
		file_class
		for file_class in list_classified_files
		if {list_files[i].top_dir_idx for i in file_class} == {0}
	]
	# files only in 1
	files_1 = [
		file_class
		for file_class in list_classified_files
		if {list_files[i].top_dir_idx for i in file_class} == {1}
	]

	total_transfer_size = sum(list_files[file_class[0]].size for file_class in files_0)
	source_size = sum(f.size for f in list_files if f.top_dir_idx == 0)
	dest_size = sum(f.size for f in list_files if f.top_dir_idx == 1)

	if display_time:
		t1 = time()
		print("B: time: {:.3f}".format(t1 - t0))
		t0 = t1

	# all temp dir names used
	# a name should not be equal to another name with a numerical prefix to avoid conflicts
	delete_tmp_dir_basename = 'temp_delete'
	move_dirs_tmp_dir_basename = 'temp_move_dirs'
	move_files_tmp_dir_basename = 'temp_move_files'
	remote_tmp_dir_basename = 'tmp_remote'

	# forbidden names for all temp dirs we have to create
	forbidden_names = set(
		[filesystems[1].name[id] for id in filesystems[1].children[0]] +
		[filesystems[0].name[id] for id in filesystems[0].children[0]])

	# select temp dir names, avoiding collisions with forbidden names
	delete_tmp_dir_name = compute_moves.gen_dir_name(delete_tmp_dir_basename, forbidden_names)
	move_dirs_tmp_dir_name = compute_moves.gen_dir_name(move_dirs_tmp_dir_basename, forbidden_names)
	move_files_tmp_dir_name = compute_moves.gen_dir_name(move_files_tmp_dir_basename, forbidden_names)
	remote_tmp_dir_name = compute_moves.gen_dir_name(remote_tmp_dir_basename, forbidden_names)

	# list empty dirs in source
	# they may need to be created or re-created in destination
	source_empty_dirs = set()
	for d, list_files_in_dirs in files[0].items():
		if len(list_files_in_dirs) == 0:
			source_empty_dirs.add(d)

	# list directories to delete. The are deleted later because they may contain
	# files or subdirectories that will be reused.
	# we move these directories to be deleted to a temp dir. The operations needing the files
	# in these directories will fetch these files from that temp dir.
	# Deletion of the dirs will be performed by deleting the temp dir.
	directories_to_delete = set()
	for d in list_dirs[1]:
		# mark dirs other than root that do not correspond to source dirs for deletion
		# root must be explicitly excluded since it is in list_dirs[1] and not in
		# dir_association_reverse
		if len(d) > 0 and not d in dir_association_reverse:
			if  len(files[1][d]) == 0 and d in source_empty_dirs:
				#optimization: do not needlessly delete and re-create empty dirs
				dir_association[d] = d
			else:
				directories_to_delete.add(d)

	# to move dirs to be deleted, reuse code from compute_moves.perform_dir_moves instead of reimplementing the same logic
	# hence add moves to dir_association and dir_association_reverse
	for d in directories_to_delete:
		new_d = (delete_tmp_dir_name,) + d
		dir_association[new_d] = d
		dir_association_reverse[d] = new_d # NB: unneeded (not used by perform_dir_moves)

	file_moves = {}
	file_copies = {}
	file_to_delete_indexes = set()

	if display_time:
		t1 = time()
		print("C: time: {:.3f}".format(t1 - t0))
		t0 = t1

	# perform the planned operations on the destination filesystem
	# this enables to check the end result
	# the logging facility of the filetree object also enables to extract the
	# list of operations performed.
	filesystems[1].set_log_state(True)

	# directory moves to produce directory correspondence
	good_associations = {
		d: dprime for (d, dprime) in dir_association.items() if not dprime == ""
	}
	compute_moves.perform_dir_moves(good_associations, filesystems[1], forbidden_names, move_dirs_tmp_dir_name)

	# determine file moves inside destination (in 'file_moves'), file copies
	# inside destination ('file_copies'), and files to delete that occur because there are
	# too many copies of them in destination (added to 'file_deletes')
	for file_class_indexes in common_files:
		file_class = [list_files[i] for i in file_class_indexes]
		# in 'file_class', all files are identical. List files in source and in
		# destination.
		s0 = set([f.relpath_tuple for f in file_class if f.top_dir_idx == 0])
		# use current paths for destination, to take into account the dir moves that were
		# performed
		s1 = set(
			[current_path(f.relpath_tuple) for f in file_class if f.top_dir_idx == 1]
		)
		# since we are processing common_files, neither s0 nor s1 is empty
		# select any source element, in case we need to do a copy
		source_elt_relpath = next(iter(s1))
		source_elt_idx = dest_idx(source_elt_relpath)
		u = s1.intersection(s0)
		#FIXME: it would be slightly simpler to take source element in u
		# don't consider elements already at the correct place
		s0 -= u
		s1 -= u
		# s0 and s1 do not have any element in common anymore
		for ref_relpath in s0:
			if len(s1) > 0:
				dest_relpath = s1.pop()
				idx = filesystems[1].path_to_node_id(dest_relpath)
				file_moves[idx] = ref_relpath
			else:
				# if we end here, s1 is empty: all elements from s1 were consumed by moves.
				# All remaining elements in s0 will be created from source_elt.
				file_copies[source_elt_idx] = ref_relpath
		# all remaining elements in s1 should be deleted
		file_to_delete_indexes |= set(
			filesystems[1].path_to_node_id(relpath) for relpath in s1
		)

	# files not appearing at all in source: will be deleted from destination.
	file_to_delete_indexes |= set(
		nid_dict[list_files[i].relpath_tuple]
		for file_class in files_1
		for i in file_class
	)

	if display_time:
		t1 = time()
		print("D: time: {:.3f}".format(t1 - t0))
		t0 = t1

	# File copies and moves inside destination
	move_file_out_of_the_way_ctr = 0
	move_file_out_of_the_way_parent_id = 0 #0 = not yet created
	def move_file_out_of_the_way(node_id, reason):
		nonlocal move_file_out_of_the_way_ctr
		nonlocal move_file_out_of_the_way_parent_id
		if move_file_out_of_the_way_parent_id == 0:
			move_file_out_of_the_way_parent_id = filesystems[1].create(0, move_files_tmp_dir_name, is_dir=True, comment = '{} - move existing file - create parent dir'.format(reason))
		filesystems[1].move(
			node_id, move_file_out_of_the_way_parent_id, new_name = str(move_file_out_of_the_way_ctr), comment = '{} - move existing file'.format(reason)
		)
		move_file_out_of_the_way_ctr += 1

    #create a path; if a file is encountered along that path, move it out of the way
	def create_dir_and_move_files_out_of_the_way(relpath, reason):
		last_id, match_length, is_dir =  filesystems[1].last_node_id_on_path(relpath)
		if match_length > 0 and not is_dir:
			move_file_out_of_the_way(last_id, reason)
		filesystems[1].create_full_path(relpath, comment = reason)

	# File copies inside destination
	copy_reason = 'file copies'
	for file_idx, relpath in file_copies.items():
		create_dir_and_move_files_out_of_the_way(relpath[:-1], reason = copy_reason)
		file_exists, fid = filesystems[1].locate(relpath)
		if file_exists:
			# copies or moves with same origin and destination should have been removed before,
			# if there is one this is a bug
			assert fid != file_idx
			# move existing file to temp dir (we can't delete it because it may be used by another copy/move operation)
			# this is correct because:
			# - the existing file cannot be at its final place
			# - all further copies or moves will work as expected
			move_file_out_of_the_way(fid, copy_reason)
		filesystems[1].copy(
			filesystems[1], file_idx, relpath[:-1], new_name = relpath[-1], comment = copy_reason
		)

	if display_time:
		t1 = time()
		print("E: time: {:.3f}".format(t1 - t0))
		t0 = t1

	# File moves inside destination
	move_reason = 'file moves'
	for file_idx, relpath in file_moves.items():
		create_dir_and_move_files_out_of_the_way(relpath[:-1], reason = move_reason)
		file_exists, fid = filesystems[1].locate(relpath)
		if file_exists:
			# copies or moves with same origin and destination should have been removed before,
			# if there is one this is a bug
			assert fid != file_idx
			# move existing file to temp dir (we can't delete it because it may be used by another copy/move operation)
			move_file_out_of_the_way(fid, move_reason)
		if sys.platform == "win32":
			# hack to work around windows bug that refuses to only change case in file name:
			# prepend a move with move_file_out_of_the_way which changes the file name into a number
			# (Windows bug confirmed on exFAT partitions, also seen in Explorer)
			origin_name = filesystems[1].name[file_idx]
			dest_name = relpath[-1]
			if (origin_name != dest_name) and (origin_name.lower() == dest_name.lower()):
				move_file_out_of_the_way(file_idx, move_reason)
		filesystems[1].move(file_idx, relpath[:-1], new_name = relpath[-1], comment = move_reason)

	# Directories to delete in destination
	# 1 - temp dir for files and dir moves
	if filesystems[1].exists(move_dirs_tmp_dir_name):
		filesystems[1].delete(move_dirs_tmp_dir_name, comment="temp dir delete")

	if filesystems[1].exists(move_files_tmp_dir_name):
		filesystems[1].delete(move_files_tmp_dir_name, comment="temp dir delete")

	# 2 - directories previously marked for deletion, which are all in delete_tmp_dir_name
	# just delete that dir
	if filesystems[1].exists(delete_tmp_dir_name):
		filesystems[1].delete(delete_tmp_dir_name, comment="temp dir delete")

	# File deletions
	for idx in file_to_delete_indexes:
		if filesystems[1].exists(idx):
			filesystems[1].delete(idx, comment="file deletions")

	if display_time:
		t1 = time()
		print("F: time: {:.3f}".format(t1 - t0))
		t0 = t1

	# save log of local operations, first part
	local_log = filesystems[1].export_and_reset_log()

	# remote full dir copies, remote actions
	for relpath in dirs_to_copy:
		if relpath not in source_empty_dirs:
			dest_relpath = (remote_tmp_dir_name,) + relpath[:-1]
			dest_parent_dir_id = filesystems[1].create_full_path(
				dest_relpath,
				comment="dir copy from source, remote part - parent creation",
			)
			filesystems[1].copy(
				filesystems[0],
				relpath,
				dest_parent_dir_id,
				comment="dir copy from source, remote part",
			)

	if display_time:
		t1 = time()
		print("G: time: {:.3f}".format(t1 - t0))
		t0 = t1

	# remote file copies, remote actions
	# we test for the existence of the file because it may have been created
	# by full dir copies
	# copy at most one file per file class
	for file_class in files_0:
		exist_files = [
			filesystems[1].exists((remote_tmp_dir_name,) + list_files[i].relpath_tuple)
			for i in file_class
		]
		if not all(exist_files):
			f = list_files[file_class[0]]
			relpath = f.relpath_tuple
			dest_relpath = (remote_tmp_dir_name,) + relpath
			dest_parent_dir_id = filesystems[1].create_full_path(
				dest_relpath[:-1],
				comment="file copy from source, remote part - parent creation",
			)
			filesystems[1].copy(
				filesystems[0],
				relpath,
				dest_parent_dir_id,
				comment="file copy from source, remote part",
			)

	remote_log = filesystems[1].export_and_reset_log()

	if display_time:
		t1 = time()
		print("H: time: {:.3f}".format(t1 - t0))
		t0 = t1

	# full dir copies, local actions
	for relpath in dirs_to_copy:
		if relpath not in source_empty_dirs:
			dest_parent_dir_id = filesystems[1].create_full_path(
				relpath[:-1],
				comment="dir copy from source, local part - parent creation",
			)
			filesystems[1].move(
				(remote_tmp_dir_name,) + relpath,
				dest_parent_dir_id,
				comment="dir copy from source, local part",
			)

	if display_time:
		t1 = time()
		print("I: time: {:.3f}".format(t1 - t0))
		t0 = t1

	# remote file copies, local actions
	# we test for the existence of the file because it may have been created by full dir copies
	# every file in a file class is re-created from the unique copy that was made in 'remote' tmp dir.
	for file_class in files_0:
		assert len(file_class) > 0
		source_relpath = (remote_tmp_dir_name,) + list_files[file_class[0]].relpath_tuple
		for i, j in enumerate(file_class):
			f = list_files[j]
			relpath = f.relpath_tuple
			if not filesystems[1].exists(relpath):
				dest_parent_dir_id = filesystems[1].create_full_path(
					relpath[:-1],
					comment="file copy from source, local part - parent creation",
				)
				if i + 1 < len(file_class):
					fid = filesystems[1].copy(
						filesystems[1],
						source_relpath,
						dest_parent_dir_id,
						new_name=relpath[-1],
						comment="file copy from source, local part",
					)
				else:
					fid = filesystems[1].move(
						source_relpath,
						dest_parent_dir_id,
						new_name=relpath[-1],
						comment="file copy from source, local part",
					)

	if display_time:
		t1 = time()
		print("J: time: {:.3f}".format(t1 - t0))
		t0 = t1

	for d in source_empty_dirs:
		filesystems[1].create_full_path(d, comment="empty dir create or re-create")

	if filesystems[1].exists(remote_tmp_dir_name):
		filesystems[1].delete(remote_tmp_dir_name, comment="temp dir delete")
	local_log += filesystems[1].export_and_reset_log()
	filesystems[1].set_log_state(False)

	ok = filesystems[1].validate() and filesystems[0] == filesystems[1]
	t1 = time()

	print("Operation sequence computation and checking time: {:.3f}s".format(t1 - tstart))
	print('Total size to transfer                  : {}'.format(misc.size_string(total_transfer_size)))
	print('Current target size                     : {}'.format(misc.size_string(dest_size)))
	print('Destination size after sync             : {}'.format(misc.size_string(source_size)))
	max_size = max(dest_size + total_transfer_size, source_size)
	print('Maximum space usage during transfer     : {}'.format(misc.size_string(max_size)))
	free_space_needed = max_size - dest_size
	print('Free space required on destination side : {}'.format(misc.size_string(free_space_needed)))
	free_space = shutil.disk_usage(destination)[2]
	print('Free space available on destination side: {}'.format(misc.size_string(free_space)))

	if free_space < free_space_needed:
		print("Not enough space to continue. Aborting.")
		sys.exit(1)

	return ok, remote_log, local_log


def analyse_pair(pair_data):
	print("Finding directory correspondence...")
	dir_moves = analyze(pair_data)
	print("Computing modifications to be applied to destination...")
	success, remote_log, local_log = compute_modifications(pair_data, dir_moves)
	operation_log = remote_log + local_log
	if not success:
		print("Internal error. Aborting.")
		sys.exit(1)
	if len(operation_log) == 0:
		print("No change to make, exiting")
		sys.exit(0)
	print("Operations to perform:")
	print("----------------------")
	print("--- remote to local operations ---")
	for o in remote_log:
		print(str(o))
	print("--- purely local operations ---")
	for o in local_log:
		print(str(o))

	ans = misc.query_yes_no(
		"Proceed with applying the {} changes?".format(len(operation_log)), "no"
	)
	if ans:
		source = pair_data["source"]
		destination = pair_data["destination"]
		effective_ops = [(str(o), o.op(source, destination)) for o in operation_log]
		for s, op in effective_ops:
			print(s)
			misc.retry(op)
		print("All changes were applied")
	else:
		print("Exiting")
		sys.exit(0)

def func_cmp(args):
	logger = logging.getLogger("main")
	source = path.abspath(args.source_dir.strip(' "'))
	destination = path.abspath(args.dest_dir.strip(' "'))
	# FIXME check that source is not within destination
	# or that destination is not within source
	if not path.isdir(source):
		print("could not stat source path {}".format(args.source_dir))
		return
	if not path.isdir(destination):
		print("could not stat destination path {}".format(args.dest_dir))
		return
	print("policy used: {}".format(args.policy))
	if args.policy == "loose":
		policy = split_policy.loose
	elif args.policy == "medium":
		policy = split_policy.medium
	elif args.policy == "strict":
		policy = split_policy.strict
	logger.info("Collecting file data...")
	ok, pair_data = traverse_trees(source, destination, policy, args.mt_tol)
	logger.info("... collection and analysis of equal files finished.")
	if not ok:
		print("Internal error. Exiting")
		sys.exit(1)
	if args.dump:
		dump_pair_data(pair_data, args.dump)
	else:
		analyse_pair(pair_data)


def dump_pair_data(pair_data, filename):
	pair_data_dump = open(filename, "wb")
	pickle.dump(pair_data, pair_data_dump, protocol=pickle.HIGHEST_PROTOCOL)
	pair_data_dump.close()


def renew_dump(pair_data):
	# used when a data structure changes, to modify existing test cases and dump them again.
	# renew_dump_add_is_dir(pair_data)
	# renew_remove_filetree(pair_data)
	# renew_change_paths(pair_data)
	# renew_change_classified_files(pair_data)
	# dump_pair_data(pair_data, 'tmp.dump')
	pass


def renew_dump_add_is_dir(pair_data):
	# removal of trailing backslashes in dirnames and the addition of
	# is_dir in filesystem objects.
	for f in pair_data["filesystems"]:
		f.is_dir = {}
		for n in f.filetree.all_nodes():
			new_name = n.tag.rstrip(path.sep)
			f.is_dir[n.identifier] = n.tag != new_name
			n.tag = new_name


def renew_remove_filetree(pair_data):
	# removal of the filetree object, replaced by a few dictionaries.
	for f in pair_data["filesystems"]:
		f.children = {}
		f.name = {}
		f.parent = {}
		f.phonebook = {}
		for n in f.filetree.all_nodes():
			name = n.tag
			id = n.identifier
			children_id = n.fpointer
			f.name[id] = name
			f.children[id] = set(children_id)
			f.phonebook[id] = {}
			for c in children_id:
				f.parent[c] = id
				f.phonebook[id][f.filetree.get_node(c).tag] = c
		for node_id in f.name:
			assert node_id == 0 or (
				node_id in f.parent and node_id in f.children[f.parent[node_id]]
			)
		del f.filetree


def renew_change_paths(pair_data):
	pair_data["list_dirs"] = [
		[filetree.filesystem.split_path(f) for f in u] for u in pair_data["list_dirs"]
	]
	pair_data["list_leaves"] = [
		[filetree.filesystem.split_path(f) for f in u] for u in pair_data["list_leaves"]
	]
	dict_fileinfo = {
		(i, p, filename): file_info(
			i, filetree.filesystem.split_path(p) + (filename,), sz, mt
		)
		for i, p, filename, sz, mt in pair_data["list_files"]
	}
	pair_data["list_files"] = list(dict_fileinfo.values())
	pair_data["files"] = [
		{filetree.filesystem.split_path(p): l for p, l in u.items()}
		for u in pair_data["files"]
	]
	pair_data["list_classified_files"] = [
		[dict_fileinfo[f] for f in fileclass]
		for fileclass in pair_data["list_classified_files"]
	]
	del pair_data["parent"]


def renew_change_classified_files(pair_data):
	d = {}
	for i, f in enumerate(pair_data["list_files"]):
		d[f] = i
	pair_data["list_classified_files"] = [
		[d[f] for f in fileclass] for fileclass in pair_data["list_classified_files"]
	]


def func_reload(args):
	print("Reloading source and destination data from dump...")
	pair_data_dump = open(args.image_file, "rb")
	pair_data = pickle.load(pair_data_dump)
	pair_data_dump.close()
	renew_dump(pair_data)
	analyse_pair(pair_data)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--logging", help="set logging level")
	parser.set_defaults(func=lambda _: parser.print_help())
	subparsers = parser.add_subparsers(help="sub-command help")
	# create the parser for the 'compare' command
	parser_cmp = subparsers.add_parser("compare", help="compare two dirs")
	parser_cmp.add_argument("source_dir", help="set source directory")
	parser_cmp.add_argument("dest_dir", help="set destination directory")
	parser_cmp.add_argument(
		"--dump", help="image file in which to dump source and destination description"
	)
	parser_cmp.add_argument("--mt_tol", type=int, default=0,
    help="toterance for file modification time difference (seconds)")
	parser_cmp.add_argument(
		"--policy",
		choices=["strict", "medium", "loose"],
		default="loose",
		help="set policy used to compare files",
	)
	parser_cmp.set_defaults(func=func_cmp)
	# create the parser for the 'reload' command
	parser_reload = subparsers.add_parser(
		"reload", help="reload image of source and dest dir"
	)
	parser_reload.add_argument("image_file", help="image file to reload")
	parser_reload.set_defaults(func=func_reload)
	args = parser.parse_args()
	if args.logging:
		numeric_level = getattr(logging, args.logging.upper(), None)
		if not isinstance(numeric_level, int):
			raise ValueError("Invalid log level: %s" % numeric_level)
		logging.basicConfig(
			level=numeric_level,
			filename="transform.log",
			format="%(asctime)s - %(message)s",
			datefmt="%Y/%d/%m %H:%M:%S",
		)
	args.func(args)
