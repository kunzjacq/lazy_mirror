from collections import defaultdict
import os
import sys
import shutil

import misc

'''
A class to represent a file tree.
Elements are either directories or regular files, and are named.
Names are unique inside a directory.
Elements can have metadata. A specific metadata for an element is identified
by a string. The set of metadata for an element is a dict.

Elements in the tree are identified with node identifiers that are integers.
These node identifiers can be obtained by path_to_node_id.
Methods that take an argument 'relpath' corresponding to a relative path in the
tree (a string, where directories are separated by os.path.sep) can also use
a node id as the relative path, or a list or tuple of dirs.

Internal functions' name begin with __. These functions do not necessarily
perform checks on their input. This is in opposition to user-facing methods
that try to prevent misuse that could lead to data structure corruption.

Logging can be turned on for tree modifications. If it is on, any operation on
the tree is recorded in self.log. These operations can later be converted to
description strings (they have a __str__ method), or performed on a real file tree
(they store the corresponding real file operation).
'''

class UnknownNodeError(Exception):
    pass


class NotADirectoryError(Exception):
    pass


class DirectoryError(Exception):
    pass


class NonexistentPathError(Exception):
    pass


class LoopError(Exception):
    pass


class FileAlreadyExistsError(Exception):
    pass


class WrongArgumentTypeError(Exception):
    pass


class RootError(Exception):
    pass


class fileOp:
    def __init__(self, name, comment):
        self.name = name
        self.comment = comment


class unaryOp(fileOp):
    def __init__(self, name, comment, op1):
        super(unaryOp, self).__init__(name, comment)
        self.op1 = op1 if len(op1) > 0 else os.path.sep

    def __str__(self):
        res = '{} {}'.format(self.name, self.op1)
        if self.comment != '':
            res = '{}: '.format(self.comment) + res
        return res


class binaryOp(fileOp):
    def __init__(self, name, comment, op1, op2):
        super(binaryOp, self).__init__(name, comment)
        self.op1 = op1 if len(op1) > 0 else os.path.sep
        self.op2 = op2 if len(op2) > 0 else os.path.sep

    def __str__(self):
        res = '{} {} to {}'.format(self.name, self.op1, self.op2)
        if self.comment != '':
            res = '{}: '.format(self.comment) + res
        return res


class deleteFileOp(unaryOp):
    def __init__(self, op1, comment=''):
        super(deleteFileOp, self).__init__('delete file', comment, op1)
        self.op = lambda source_root, dest_root: \
					lambda: os.remove(os.path.join(dest_root, op1))


class deleteDirOp(unaryOp):
    def __init__(self, op1, comment=''):
        super(deleteDirOp, self).__init__('delete dir', comment, op1)
        self.op = lambda source_root, dest_root: \
					lambda: shutil.rmtree(os.path.join(dest_root, op1))


class createDirOp(unaryOp):
    def __init__(self, op1, comment=''):
        super(createDirOp, self).__init__('create dir', comment, op1)
        self.op = lambda source_root, dest_root: \
					lambda: os.makedirs(os.path.join(dest_root, op1))


class createFileOp(unaryOp):
    def __init__(self, op1, comment=''):
        super(createFileOp, self).__init__('create file', comment, op1)
        # there no actual operation that corresponds to the creation of a file
        self.op = lambda source_root, dest_root: \
					lambda:  None


class moveDirOp(binaryOp):
    def __init__(self, op1, op2, comment=''):
        super(moveDirOp, self).__init__('move dir', comment, op1, op2)
        self.op = lambda source_root, dest_root: \
					lambda: misc.move(os.path.join(dest_root, op1), os.path.join(dest_root, op2))


class moveFileOp(binaryOp):
    def __init__(self, op1, op2, comment=''):
        super(moveFileOp, self).__init__('move file', comment, op1, op2)
        self.op = lambda source_root, dest_root: \
					lambda: misc.move(os.path.join(dest_root, op1), os.path.join(dest_root, op2))


class localCopyFileOp(binaryOp):
    def __init__(self, op1, op2, comment=''):
        super(localCopyFileOp, self).__init__(
            'local file copy', comment, op1, op2)
        self.op = lambda source_root, dest_root: \
					lambda: shutil.copy2(os.path.join(dest_root, op1), os.path.join(dest_root, op2))


class remoteCopyFileOp(binaryOp):
    def __init__(self, op1, op2, comment=''):
        super(remoteCopyFileOp, self).__init__(
            'remote file copy', comment, op1, op2)
        self.op = lambda source_root, dest_root: \
					 lambda: shutil.copy2(os.path.join(source_root, op1), os.path.join(dest_root, op2))


class localCopyDirOp(binaryOp):
    def __init__(self, op1, op2, comment=''):
        super(localCopyDirOp, self).__init__(
            'local dir copy', comment, op1, op2)
        self.op = lambda source_root, dest_root: \
				 lambda: shutil.copytree(os.path.join(dest_root, op1), os.path.join(dest_root, op2))


class remoteCopyDirOp(binaryOp):
    def __init__(self, op1, op2, comment=''):
        super(remoteCopyDirOp, self).__init__(
            'remote dir copy', comment, op1, op2)
        self.op = lambda source_root, dest_root: \
					lambda: shutil.copytree(os.path.join(source_root, op1), os.path.join(dest_root, op2))

class filesystem:
    def __init__(self):
        self.curr_id = 0
        self.logActions = False
        self.log = []
        self.metadata = defaultdict(dict)
        self.is_dir = {}  # indicate, for each node id, if it is a directory
        self.children = {}  # children[node_id] = set of ids of children
        # must be empty if is_dir[node_id] = False
        self.name = {} # name[node_id] file/dir name
        self.parent = {}  # parent[node_id] = id of parent
        # phonebook[node_id][child_name] = id of child of node_id with name 'child_name'
        self.phonebook = {}
        self.__add_node(0, '', True)

    def __eq__(self, other):
        '''
          Tests whether two filesystems are equal, i.e. whether the filename and
          metadata tree are equal. node ids for the identical files do not have to be
          the same for file trees to be considered equal.
        '''
        if not isinstance(other, filesystem):
            raise WrongArgumentTypeError(
                'filesystem.equals expects a second argument of type filesystem')
        if other is self:
            return True
        return self.equals_from_node(other, 0, 0)

    def equals_from_node(self, other, node_id_1, node_id_2):
        '''
        Auxiliary recursive method for tree comparison.
        There is no need to compare tags of input nodes:
        - at the root level, they are equal (empty tag)
        - at levels below root, they are compared by the calling function
          (function invariant)
        '''
        if self.metadata[node_id_1] != other.metadata[node_id_2]:
            print('difference in METADATA at node id {}, name \'{}\''.format(
                node_id_1, self.name[node_id_1]))
            return False
        if self.is_dir[node_id_1] != other.is_dir[node_id_2]:
            print('difference in DIR STATUS at node id {}, name \'{}\''.format(
                node_id_1, self.name[node_id_1]))
            return False
        children1 = self.phonebook[node_id_1]
        children2 = other.phonebook[node_id_2]
        if children1.keys() != children2.keys():
            print('difference in CHILDREN at node id {}, name \'{}\''.format(
                node_id_1, self.name[node_id_1]))
            print(children1.keys())
            print(children2.keys())
            print(set(children1.keys()).difference(set(children2.keys())))
            print(set(children2.keys()).difference(set(children1.keys())))
            return False
        for name in children1:
            if not self.equals_from_node(other, children1[name], children2[name]):
                return False
        return True

    @staticmethod
    def check_relpath_type(relpath):
        if not isinstance(relpath, int) and not isinstance(relpath, str) \
            and not ((isinstance(relpath, list) or isinstance(relpath, tuple))
                     and all(isinstance(d, str) for d in relpath)):
            raise WrongArgumentTypeError('method need a relpath type')

    @staticmethod
    def split_path(p):
        if (isinstance(p, list) or isinstance(p, tuple)) and \
                (len(p) == 0 or isinstance(p[0], str)):
            return tuple(d for d in p if len(d) > 0 and d != '.')
        elif isinstance(p, str):
            return filesystem.__split_path(p)
        else:
            raise WrongArgumentTypeError

    @staticmethod
    def __split_path(p):
        npath = os.path.normpath(p)
        return tuple(d for d in npath.split(os.path.sep) if len(d) > 0 and d != '.')

    @staticmethod
    def assemble_path(dir_list):
        if isinstance(dir_list, str):
            return str
        elif (isinstance(dir_list, list) or isinstance(dir_list, tuple)) and \
                (len(dir_list) == 0 or isinstance(dir_list[0], str)):
            return filesystem.__assemble_path(dir_list)
        else:
            raise WrongArgumentTypeError

    @staticmethod
    def __assemble_path(dir_list):
        return os.path.sep.join([d for d in dir_list if len(d) > 0 and d != '.'])

    def __check_node_id(self, node_id):
        if not isinstance(node_id, int):
            raise WrongArgumentTypeError('an int is expected as node id')
        if not node_id in self.name:
            raise UnknownNodeError(
                'node {} does not belong to filesystem'.format(node_id))

    def __remove_node(self, node_id, clear_parent=True):
        if clear_parent:
            parent_id = self.parent[node_id]
            self.children[parent_id].remove(node_id)
            del self.phonebook[parent_id][self.name[node_id]]
        del self.parent[node_id]
        del self.children[node_id]
        del self.phonebook[node_id]
        del self.name[node_id]
        del self.is_dir[node_id]
        if node_id in self.metadata:
            del self.metadata[node_id]

    def __remove_subtree(self, node_id, clear_parent=True):
        for c in self.children[node_id]:
            self.__remove_subtree(c, False)
        self.__remove_node(node_id, clear_parent)

    def __move_node(self, node_id, parent_id, new_name=None):
        old_parent_id = self.parent[node_id]
        if parent_id != old_parent_id:
            self.parent[node_id] = parent_id
            self.children[old_parent_id].remove(node_id)
            self.children[parent_id].add(node_id)
        n = self.name[node_id]
        if parent_id != old_parent_id or (not new_name is None and n != new_name):
            np = n if new_name is None else new_name
            del self.phonebook[old_parent_id][n]
            # ensure that the move does not create identical names in target dir.
            assert(np not in self.phonebook[parent_id])
            self.phonebook[parent_id][np] = node_id
            if n != np:
                self.name[node_id] = np

    def __add_node(self, parent_id, name, is_dir):
        node_id = self.curr_id
        self.curr_id += 1
        if node_id > 0:
            #assert(name not in self.phonebook[parent_id])
            if name in self.phonebook[parent_id]:
                raise FileAlreadyExistsError
            self.parent[node_id] = parent_id
            self.children[parent_id].add(node_id)
            self.phonebook[parent_id][name] = node_id
        self.phonebook[node_id] = {}
        self.children[node_id] = set()
        self.name[node_id] = name
        self.is_dir[node_id] = is_dir
        return node_id

    def __depth(self, node_id):
        res = 0
        if node_id == 0:
            return -1
        while self.parent[node_id] != 0:
            node_id = self.parent[node_id]
            res = res + 1
        return res

    def __walk_downwards(self, relpath, on_success, on_failure, create_dir=False, comment=''):
        '''
        If a file with path 'relpath' exists,
        executes 'on_success(id)', with 'id' the node_id of node with path 'relpath'.
        otherwise, executes 'on_failure(relpath)'.
        This functions accepts node ids instead of paths in 'relpath'.
        In that case, and if the node id exists in the tree,
        executes 'on_success(relpath)' directly.
        '''
        if isinstance(relpath, int):
            return on_success(relpath, self.__depth(relpath)) if relpath in self.name else on_failure(relpath, 0, 0)
        else:
            relpath = filesystem.split_path(relpath)
        curr_id = 0
        for i, d in enumerate(relpath):
            if d in self.phonebook[curr_id]:
                # all nodes traversed, except maybe the last one, must be directories
                curr_id = self.phonebook[curr_id][d]
            else:
                if create_dir:
                    curr_id = self.create(
                        curr_id, d, comment=comment, is_dir=True)
                else:
                    return on_failure(relpath, curr_id, i)
        return on_success(curr_id, len(relpath))

    def validate(self):
        '''
        Performs consistency checks on the tree: ensure that
        - names are unique in each dir
        - for every n in children[p], phonebook[name[n]] = n
        - only dirs have children
        - node belongs to children[parent[node]]
        - for every n in children[p], parent[n] = p
          this ensures that a node idx does not appear at multiple places in
          the tree, and therefore that there is no loop in the tree
        - for every p, phonebook[p].values() = children[p]
        '''
        return self.validate_subtree(0)

    def validate_subtree(self, root_id):
        children_names = self.phonebook[root_id].keys()
        if set(self.phonebook[root_id].values()) != self.children[root_id] or \
            (not self.is_dir[root_id] and len(self.children[root_id]) > 0) or \
            (len(children_names) != len(set(children_names))) or \
                (root_id != 0 and root_id not in self.children[self.parent[root_id]]):
            return False
        for c in self.children[root_id]:
            if self.phonebook[root_id][self.name[c]] != c or self.parent[c] != root_id:
                return False
        for c in self.children[root_id]:
            if not self.validate_subtree(c):
                return False
        return True

    def is_parent(self, node_id_1, node_id_2):
        self.__check_node_id(node_id_1)
        self.__check_node_id(node_id_2)
        return self.__is_parent(node_id_1, node_id_2)

    def test_is_dir(self, relpath):
        node_id = self.path_to_node_id(relpath)
        return self.is_dir[node_id]

    def __is_parent(self, node_id_1, node_id_2):
        # tests whether node_id_1 is on the path from node_id_2 to the root,
        # including if node_id_1 == node_id_2
        if node_id_2 == node_id_1:
            return True
        while node_id_2 != 0:
            node_id_2 = self.parent[node_id_2]
            if node_id_2 == node_id_1:
                return True
        return False

    def node_path(self, node_id):
        self.__check_node_id(node_id)
        return self.__node_path(node_id)

    def __node_path(self, node_id):
        path = []
        while node_id != 0:
            path_elt = self.name[node_id]
            path.insert(0, path_elt)
            node_id = self.parent[node_id]
        return tuple(path)

    # returns the node id corresponding to a path, or a
    # NonexistentPathError exception if no node exists with that path.
    def path_to_node_id(self, relpath):
        filesystem.check_relpath_type(relpath)
        def on_failure(relpath, last_id, match_length):
            raise NonexistentPathError(
                'Path {} does not exist'.format(relpath))
        return self.__walk_downwards(relpath, lambda curr_id, match_length : curr_id, on_failure)

    # finds the deepest node on a path. returns the id of that node,
    # the size of the subpath of relpath it corresponds to, and its type (dir or not)
    def last_node_id_on_path(self, relpath):
        filesystem.check_relpath_type(relpath)
        def on_failure(relpath, last_id, match_length):
            return (last_id, match_length, self.is_dir[last_id])
        return self.__walk_downwards(
            relpath, lambda curr_id, match_length: (curr_id, match_length, self.is_dir[curr_id]),
            on_failure)

    def locate(self, relpath):
        filesystem.check_relpath_type(relpath)
        return self.__walk_downwards(
            relpath, lambda curr_id, match_length: (True,  curr_id),
            lambda relpath, last_id, match_length: (False, 0))

    def exists(self, relpath):
        b, _ = self.locate(relpath)
        return b

    def child_exists_error(self, parent_id, childname, relpath=None):
        if childname in self.phonebook[parent_id]:
            path_string = self.__relpath_to_str(
                relpath) if not relpath is None else self.__node_path(parent_id)
            raise FileAlreadyExistsError("node %s already exists in path %s"
                                         % (childname, path_string))

    def relpath_to_str(self, relpath):
        if isinstance(relpath, int):
            return filesystem.__assemble_path(self.node_path(relpath))
        elif isinstance(relpath, str):
            assert(self.exists(relpath))
            return relpath
        if isinstance(relpath, list) or isinstance(relpath, tuple):
            assert(self.exists(relpath))
            return filesystem.__assemble_path(relpath)
        else:
            raise WrongArgumentTypeError(
                'type {} illegal as relpath_to_str input'.format(type(relpath)))

    def __relpath_to_str(self, relpath):
        if isinstance(relpath, int):
            return filesystem.__assemble_path(self.__node_path(relpath))
        elif isinstance(relpath, str):
            return relpath
        elif isinstance(relpath, list) or isinstance(relpath, tuple):
            return filesystem.__assemble_path(relpath)
        else:
            assert(0)

    def add_metadata(self, relpath, key, value):
        filesystem.check_relpath_type(relpath)

        def report_error(_):
            raise NonexistentPathError(
                "Path or node id %s does not exist" % relpath)

        def write_metadata(nid, key, value):
            self.metadata[nid][key] = value
        if not isinstance(key, str):
            raise WrongArgumentTypeError('metadata keys must be strings')
        self.__walk_downwards(
            relpath, lambda nid, match_length: write_metadata(nid, key, value), report_error)

    def print_metadata(self):
        for x in self.metadata:
            for y in self.metadata[x]:
                print('{}[{}]={}'.format(
                    y, self.__node_path(x), self.metadata[x][y]))

    def create(self, parent_relpath, name, is_dir=False, comment=''):
        parent_id = self.path_to_node_id(parent_relpath)
        if not isinstance(name, str):
            raise WrongArgumentTypeError('Name must be a strings')
        node_id = self.__add_node(parent_id, name.rstrip(os.path.sep), is_dir)
        if self.logActions:
            s = os.path.join(self.__relpath_to_str(parent_relpath), name)
            op = createDirOp if is_dir else createFileOp
            self.log.append(op(s, comment=comment))
        return node_id

    # equivalent of mkdir -p
    # returns the id of the last created node
    def create_full_path(self, relpath, comment=''):
        filesystem.check_relpath_type(relpath)
        return self.__walk_downwards(relpath, lambda dir_id, depth: dir_id, lambda _: True, True, comment=comment)

    def delete(self, relpath, comment=''):
        '''
        removes file or dir of path 'relpath', and all nodes below it.
        path can be a string of dirs sepated by os.path.sep, a descending list of
        dirs, or a node id.
        throws a NonExistentPath error if relpath is not found in the tree.
        '''
        node_id = self.path_to_node_id(relpath)
        if(node_id == 0):
            raise RootError('Cannot delete root')
        if self.logActions:
            op = deleteDirOp if self.is_dir[node_id] else deleteFileOp
            self.log.append(
                op(self.__relpath_to_str(relpath), comment=comment))
        self.__remove_subtree(node_id)

    def move(self, old_relpath, new_parent_relpath, new_name=None, comment=''):
        '''
        move 'old_path' into dir 'new_path':
        if 'old_relpath' ends with file or dir d, d is moved into 'new_parent_relpath'.
        a path can be a string of dirs sepated by os.path.sep, a descending list of
        dirs, or a node id.
        'old_relpath' can be a directory or a file; 'new_parent_relpath' must be a directory.
        optionally renames 'd' into 'new_name'.
        '''
        if not new_name is None:
            assert(isinstance(new_name, str))
        node_id = self.path_to_node_id(old_relpath)
        self_parent_id = self.path_to_node_id(new_parent_relpath)
        if node_id == 0:
            raise RootError('cannot move root')
        name = new_name if not new_name is None else self.name[node_id]
        if self.__is_parent(node_id, self_parent_id):
            raise LoopError("cannot move %s to %s as this would create a loop"
                            % (self.__relpath_to_str(old_relpath),
                               os.path.join(self.__relpath_to_str(new_parent_relpath), name)))
        if not self.is_dir[self_parent_id]:
            raise NotADirectoryError(
                "target node is not a directory: cannot move anything into it")
        if self.logActions:
            origin = self.__relpath_to_str(old_relpath)
        # ensure that the move does not create duplicate names in target dir.
        self.child_exists_error(self_parent_id, name)
        self.__move_node(node_id, self_parent_id, new_name)
        if self.logActions:
            dest = os.path.join(
                self.__relpath_to_str(new_parent_relpath), name)
            op = moveDirOp if self.is_dir[node_id] else moveFileOp
            self.log.append(op(origin, dest, comment=comment))

    def copy(self, other, other_relpath, new_parent_relpath, new_name=None,
             copy_metadata_if_different_fs=True, comment=''):
        '''
        copies file or dir with path 'other_relpath' in filesystem 'other' into
        'new_parent_relpath'. Optionally renames element, giving it name 'new_name'.
        If copied element is a directory, recursively copy children.
        '''
        if not new_name is None and not isinstance(new_name, str):
            raise WrongArgumentTypeError('new_name must be a string')
        if not self is other and not isinstance(other, filesystem):
            raise WrongArgumentTypeError('\'other\' must be a filesystem')
        other_id = other.path_to_node_id(other_relpath)
        other_name = other.name[other_id]
        self_parent_id = self.path_to_node_id(new_parent_relpath)
        if not self.is_dir[self_parent_id]:
            raise NotADirectoryError(
                "copy_file(): destination node is not a directory: " +
                "cannot copy anything into it")
        name = other_name if new_name is None else new_name
        # ensure that the copy does not create duplicate names in target dir.
        self.child_exists_error(self_parent_id, name)
        copy_metadata = self is other or copy_metadata_if_different_fs
        new_node_id = self.__copy_subtree(other, other_id, self_parent_id,
                                          copy_metadata, name)
        if self.logActions:
            origin = other.__relpath_to_str(other_id)
            dest = os.path.join(
                self.__relpath_to_str(new_parent_relpath), name)
            if other.is_dir[other_id]:
                op = localCopyDirOp if other is self else remoteCopyDirOp
            else:
                op = localCopyFileOp if other is self else remoteCopyFileOp
            self.log.append(op(origin, dest, comment=comment))
        return new_node_id

    def __copy_subtree(self, other, other_id, self_parent_id, copy_metadata, name=None):
        if name is None:
            name = other.name[other_id]
        new_node_id = self.__add_node(
            self_parent_id, name, other.is_dir[other_id])
        if copy_metadata and other_id in other.metadata:
            self.metadata[new_node_id] = dict(
                other.metadata[other_id])  # copy metadata
        for child_id in other.children[other_id]:
            self.__copy_subtree(other, child_id, new_node_id, copy_metadata)
        return new_node_id

    def show(self):
        # FIXME: don't have a show method anymore!
        pass

    def set_log_state(self, b):
        '''
        Enable or disable (depending on boolean 'b') the logging of actions
        performed on the filesystem.
        '''
        self.logActions = b

    def export_and_reset_log(self):
        l = self.log
        self.log = []
        return l


def base_fs():
    f = filesystem()
    f.create('', 'u', is_dir=True)
    f.create('u', 'v', is_dir=True)
    f.create('u', 'w', is_dir=True)
    f.create('u' + os.path.sep + 'v', 'txt1', is_dir=False)
    return f


def add_and_move_1_t():
    f = base_fs()
    g = filesystem()
    g.create('', 'u', is_dir=True)
    g.create('u', 'w', is_dir=True)
    g.create('u' + os.path.sep + 'w', 'v', is_dir=True)
    g.create('u' + os.path.sep + 'w' + os.path.sep + 'v', 'txt1', is_dir=False)
    h = base_fs()
    f.move('u' + os.path.sep + 'v', 'u' + os.path.sep + 'w')
    result1 = f == g
    result2 = f == h
    f.move('u' + os.path.sep + 'w' + os.path.sep + 'v', 'u')
    result3 = f == h
    return result1 and not result2 and result3


def add_and_move_2_t():
    f = base_fs()
    g = filesystem()
    g.create('', 'u', is_dir=True)
    g.create(['u'], 'w', is_dir=True)
    g.create(['u', 'w'], 'v', is_dir=True)
    g.create(['u', 'w', 'v'], 'txt1', is_dir=False)
    h = base_fs()
    f.move(['u', 'v'], ['u', 'w'])
    result1 = f == g
    result2 = f == h
    f.move(['u', 'w', 'v'], 'u')
    result3 = f == h
    return result1 and not result2 and result3


def track_and_move_t():
    f = base_fs()
    # memorize node id of u\v\txt1 before moving the file
    node_id = f.path_to_node_id(['u', 'v', 'txt1'])
    f.move(['u', 'v'], ['u', 'w'])
    p = os.path.sep.join(f.node_path(node_id))
    p_exp = os.path.sep.join(['u', 'w', 'v', 'txt1'])
    res1 = p == p_exp
    if not res1:
        print('expected {}, got {}'.format(p_exp, p))
    f = base_fs()
    # memorize node id of u\v\txt1 before moving the file
    node_id = f.path_to_node_id(['u', 'v', 'txt1'])
    f.move(['u', 'v'], ['u', 'w'], 'v2')
    q = os.path.sep.join(f.node_path(node_id))
    q_exp = os.path.sep.join(['u', 'w', 'v2', 'txt1'])
    res2 = q == q_exp
    if not res2:
        print('expected {}, got {}'.format(q_exp, q))
    return res1 and res2


def file_already_exists_error_1_t():
    try:
        f = base_fs()
        f.create('u' + os.path.sep + 'v', 'txt1', is_dir=False)
    except FileAlreadyExistsError:
        # e,b,t = sys.exc_info()
        # print(str(b))
        return True
    return False


def file_already_exists_error_2_t():
    try:
        f = base_fs()
        f.create(['u', 'v'], 'txt1', is_dir=False)
    except FileAlreadyExistsError:
        # e,b,t = sys.exc_info()
        # print(str(b))
        return True
    return False


def wrong_argument_type_error_t():
    try:
        f = base_fs()
        f.create((0, 1), 'txt1', is_dir=False)
    except WrongArgumentTypeError:
        # e,b,t = sys.exc_info()
        # print(str(b))
        return True
    return False


def loop_error_t():
    try:
        f = base_fs()
        f.move(['u'], ['u', 'v'])
    except LoopError:
        # e,b,t = sys.exc_info()
        # print(str(b))
        return True
    return False


def not_a_directory_error_t():
    try:
        f = base_fs()
        f.move(['u', 'w'], ['u', 'v', 'txt1'])
    except NotADirectoryError:
        # e,b,t = sys.exc_info()
        # print(str(b))
        return True
    return False


def delete_file_t():
    f = base_fs()
    f.delete(['u', 'v'])
    g = filesystem()
    g.create('', 'u', is_dir=True)
    g.create(['u'], 'w', is_dir=True)
    return f == g


def delete_root_error_t():
    f = base_fs()
    try:
        f.delete([])
    except RootError:
        # e,b,t = sys.exc_info()
        # print(str(b))
        return True
    return False


def exists_t():
    f = base_fs()
    r1 = f.exists(['u'])
    r2 = f.exists(['u', 'v'])
    r3 = f.exists(['u', 'v', 'txt1'])
    r4 = f.exists(['v'])
    r5 = f.exists(['u', 'txt1'])
    r6 = f.exists(['u', 'h'])
    # print('{} {} {} {} {} {}'.format(r1, r2, r3, not r4, not r5, not r6))
    return r1 and r2 and r3 and not r4 and not r5 and not r6


def create_full_path_t():
    f = base_fs()
    f.create_full_path(['u', 'v', 'w', 'x', 'y', 'z'])
    return f.exists(['u', 'v', 'w', 'x', 'y', 'z'])


def copy_subtree_t():
    f = base_fs()
    # copy u/v into u/w
    f.copy(f, ['u', 'v'], ['u', 'w'])
    g = base_fs()
    g.create(['u', 'w'], 'v', is_dir=True)
    g.create(['u', 'w', 'v'], 'txt1', is_dir=False)
    return f == g


def run_tests():
    test_list = [add_and_move_1_t, add_and_move_2_t, track_and_move_t, delete_file_t,
                 exists_t, create_full_path_t, file_already_exists_error_1_t,
                 file_already_exists_error_2_t, wrong_argument_type_error_t, loop_error_t,
                 not_a_directory_error_t, delete_root_error_t, copy_subtree_t]
    results = []
    for i, t in enumerate(test_list):
        test_result = t()
        print('{}: {}\n {}'.format(i+1, t.__name__,
                                   'passed' if test_result else 'failed'))
        results.append(test_result)
    return all(results)
