# lazy_mirror
 mirroring tool with lazy file comparison logic

A python3 script for path mirrorring, that attempts to minimize file transfers
between reference (source) and destination dir. It is useful for instance to 
synchronize a local dir with a network share.

There is NO WARRANTY WHATSOEVER regarding the correct behavior of this script.
Use with caution, and review the changes proposed before accepting them (the
script gives the user the opportunity to do that).

The file transfers between source and destination are minimized by ensuring that 
all the files in source that are available on the destination side are copied 
or moved inside destination instead of being transferred from source.

If some files were renamed or reoragnized in source after a previous
synchronization for instance, this enables to propagate these renames without
any file copy from source to destination.

The scripts uses a user-selectable logic to determine which files are equal in source 
and destination. The default logic is loose and only relies on file size and 
modification time, but ignores file names.
It may therefore produce false-positives (files wrongly assumed to be equal)
and false-negatives. Use of stricter logic ruling out false-positives is possible,
(see 'medium' and 'strict' settings), but negates most of the benefits of this tool,
since these stricter settings require accessing the contents of possibly remote
files.

False-positives, false-negatives or failure to complete operations may lead to
an incoherent state with files being deleted on the destination side, but in
such a case the source side is still available to change the comparison logic
or retry the operation.

False positives may lead to a seemingly complete transfer where destination
is in fact different from source. False negatives on the contrary lead to a
correct	end-state, but with unneeded file transfers.

There is no provision for the management of symbolic links.

Full documentation is in mirror.py.