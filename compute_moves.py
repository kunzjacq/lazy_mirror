import logging
from os import path

import filetree

def decompose_path(relpath):
	'''
	decompose a path into its last part and the rest
	'''
	split_path = filetree.filesystem.split_path(relpath)
	first_part = filetree.filesystem.assemble_path(split_path[:-1])
	if len(split_path) > 0:
		last_part = split_path[-1]
	else:
		last_part = ''
	return first_part, last_part

def gen_dir_name(prefix, forbidden_names):
	temp_dir_name = prefix
	if temp_dir_name in forbidden_names:
		ctr = 0
		while temp_dir_name + '_' + str(ctr) in forbidden_names:
			ctr += 1
		temp_dir_name += '_'  + str(ctr)
	return temp_dir_name


def perform_dir_moves(dir_association, target_fs, forbidden_names, temp_dir_name):
	'''
	computes and perform a series of moves on the filesystem target_fs = fs[1] in order to have 
	for every pair of paths a,b with dir_association[a] = b,
	b moved to a after all moves were performed.
	Returns tmp_path, the name of a temp directory used for the moves, that 
	may or may not have been created during the operation of the function,
	that may contain some remaining files, and that is not deleted by
	perform_dir_moves.
	'''
	logger = logging.getLogger(__name__)
	def top_dir(p):
		return filetree.filesystem.split_path(p)[0]
	node_idx_map = {}
	# list all subdirs of root; root dir always has id 0
	toplevel_names_target_all = [target_fs.name[id] for id in target_fs.children[0]]
	toplevel_names_target_dirs = [target_fs.name[id] for id in target_fs.children[0] if target_fs.is_dir[id]]

	# the ordering below ensures that parents are processed before children
	ordered_targets = sorted([a for a in dir_association], key = lambda r:len(r))
	for a in ordered_targets:
		b = dir_association[a]
		try:
			node_idx_map[a] = target_fs.path_to_node_id(b)
		except NonexistentPathError:
			logger.error('perform_dir_moves(): non-existent move source, aborting')
			raise

	# before doing the actual work, move everything into temp dir, 
	# in order to avoid conflicts between fs and destination dirs
	for b in toplevel_names_target_all:
		if (b,) in dir_association and dir_association[(b,)] == (b,):
			# optimization: do not move to temp dir a directory at toplevel that is to be moved
			# if its final location is equal to its initial location. 
			# Indeed, it would be moved back in place at next step anyway.
			continue
		if not b in toplevel_names_target_dirs and (b,) not in dir_association:
			# optimization: do not move toplevel files if they are not the target of a dir move
			# does not take into account the case where they are in the way of a directory 
			# creation for a file move. However, this case is handled separately 
			# (see file copies / moves)
			continue
		temp_dir_id = target_fs.create_full_path((temp_dir_name,))
		target_fs.move((b,), temp_dir_id, comment = 'dir shuffle, move toplevel dirs or files to dir move temp dir')

	for a in ordered_targets:
		dest_pathlist = a
		source_pathlist = target_fs.node_path(node_idx_map[a])
		if dest_pathlist == source_pathlist:
			# source and destination are the same path, nothing to do
			continue

		exists_dest_parent, dest_parent_id = target_fs.locate(dest_pathlist[:-1])
		exists_leaf, leaf_id = target_fs.locate(dest_pathlist)
		dest_is_source_subdir = dest_pathlist[:len(source_pathlist)] == source_pathlist

		if dest_is_source_subdir:
			# destination is a subdir of source: need to move source to temp_dir first
			tmp_subpath_id = target_fs.create_full_path((temp_dir_name,) + source_pathlist[:-1])
			target_fs.move(node_idx_map[a], tmp_subpath_id, comment='dir shuffle, dest is parent of source case')

		if not exists_dest_parent or dest_is_source_subdir:
			dest_parent_id = target_fs.create_full_path(dest_pathlist[:-1], comment='dir shuffle, create parent of target dir')
		elif exists_leaf:
			tmp_subpath_id = target_fs.create_full_path((temp_dir_name,) + dest_pathlist[:-1])
			target_fs.move(leaf_id, tmp_subpath_id, comment='dir shuffle, free location for target dir by moving file already there')

		target_fs.move(node_idx_map[a], dest_parent_id, dest_pathlist[-1], comment='dir shuffle, final move')
	return

def compute_moves_t():
	def build_move_map(target_fs, dir_association):
		node_idx_map = {}
		for a in dir_association:
			b = dir_association[a]
			try:
				node_idx_map[a] = target_fs.path_to_node_id(b)
			except:
				print('Non-existent move source')
				raise SystemExit(1)
		return node_idx_map

	def check_dir_moves(dir_association, target_fs, node_idx_map):
		result = True
		verbose = False
		for a in dir_association:
			b = dir_association[a]
			ar = filetree.filesystem.assemble_path(a)
			if verbose: print('{} -> {}:'.format(b,ar))
			real_a = filetree.filesystem.assemble_path(target_fs.node_path(node_idx_map[a]))
			if verbose:
				if ar == real_a:
					print('{} was moved to {} as expected'.format(b, ar))
			if ar != real_a:
				result = False
				if verbose: print('{} should be in {}, is in {}'.format(b, ar, real_a))
		return result

	def run_test(paths_to_create, dir_correspondence):
		f = filetree.filesystem()
		for p in paths_to_create:
			f.create_full_path(p)
		source_map = build_move_map(f, dir_correspondence)
		compute_dir_moves(dir_correspondence, f)
		return check_dir_moves(dir_correspondence, f, source_map)
	# test 1
	result1 = run_test(
		paths_to_create = [['b', '1'],['c', '2']],
		dir_correspondence = {('a', 'b'): 'b', ('a',) : 'c'})
	# test 2
	result2 = run_test(
		paths_to_create = [['a', '1'],['b', '2']],
		dir_correspondence = {('a', 'c'): 'b', ('b','d') : 'a'})
	# test 3, triggers leaf move
	result3 = run_test(
		paths_to_create = [['a', 'b', '1'],['b', '2']],
		dir_correspondence = {('a',): 'a', ('a', 'b') : 'b'})
	return result1 and result2 and result3
