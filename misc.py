import sys, os

def query_yes_no(question, default="yes"):
	"""
	code from stackoverflow 3041986
	Ask a yes/no question via raw_input() and return their answer.

	"question" is a string that is presented to the user.
	"default" is the presumed answer if the user just hits <Enter>.
	It must be "yes" (the default), "no" or None (meaning
	an answer is required of the user).

	The "answer" return value is True for "yes" or False for "no".
	"""
	valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
	if default is None:
		prompt = " [y/n] "
	elif default == "yes":
		prompt = " [Y/n] "
	elif default == "no":
		prompt = " [y/N] "
	else:
		raise ValueError("invalid default answer: '%s'" % default)

	while True:
		sys.stdout.write(question + prompt)
		try:
			choice = input().lower().rstrip()
		except EOFError:
			choice = ""
		if default is not None and choice == "":
			return valid[default]
		elif choice in valid:
			return valid[choice]
		else:
			sys.stdout.write("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")

def size_string(n, precision=1):
		"""
		Convert n into a human-readable string representing a byte size.
		precision: number of digits after point for sizes above 1 KB.
		"""
		n = int(n)
		symbols = ('B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB')
		if n < (1 << 10):
			return '{} {}'.format(n, symbols[0])
		fmt_string = '{{:.{}f}} {{}}'.format(precision)
		value = float(n)
		i = 0
		while value >= (1<<10) and i + 1 < len(symbols):
			value /= 1 << 10
			i += 1
		return fmt_string.format(value, symbols[i])

def move(src, dst):
		"""Logic from shutils, but simplified.
		We NEVER want to do a copy when the move fails as in shutils (shutils does not
		check the cause of the error and just seems to assume that it is because source
		and destination are on different filesystems).
		Indeed, a move may fail because of permission issues (or if a file is locked
		by a process under Windows). This is not a valid reason to do a copy. In such a case,
		let 'os' raise an exception.
		"""
		def samefile(src, dst):
				if hasattr(os.path, 'samefile'):
						try:
								return os.path.samefile(src, dst)
						except OSError:
								return False
				return (os.path.normcase(os.path.abspath(src)) ==
								os.path.normcase(os.path.abspath(dst)))
		def basename(path):
			sep = os.path.sep + (os.path.altsep or '')
			return os.path.basename(path.rstrip(sep))
		real_dst = dst
		if os.path.isdir(dst) and not samefile(src, dst):
			#move src in dst, do not rename src as dst
			real_dst = os.path.join(dst, basename(src))
		os.rename(src, real_dst)

def retry(op):
	done = False
	while not done:
		try:
			op()
			done = True
		except Exception as e:
			print(e)
			if not query_yes_no("Retry?", "no"):
				print('Aborting')
				exit
