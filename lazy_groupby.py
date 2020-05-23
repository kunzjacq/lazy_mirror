def lazy_groupby(elts, splitters, class_is_final):
		'''
		lazy_groupby computes an equivalence class relation between objects 'elts'.
		Each splitter is a tuple
		(attribute_name, attribute_computation_function, splitting_function)
		the attribute_computation_function takes an integer, referencing an object
		whose attribute can be computed, and returns the corresponding attribute value.
		The splitting function takes as input a list of (integer, attribute value) and
		separates the corresponding objects into distinct classes. It produces a list of
		integer lists, each integer list representing an object class.
		The simplest splitting logic is to put objects with distinct attribute values
		into distinct	classes. The mechanism by which objects are associated to integers
		is outside the	scope	of lazy_groupby and handled by splitters and class_is_final
		directly.
		Each splitter is computed successively on all not yet classified elements,
		and elements are split according to the corresponding splitting function logic.
		Before the next attribute is computed, each object class is passed alongside the
		list of attributes already used to 'class_is_final' which decides whether the class
		should be processed for	further splitting or not. This is done to avoid having to
		perform expensive	attribute computations if it is not needed (e.g. if the class size
		is 1,	i.e. it can't	be split further, there is no need to compute more attributes).
		'''
		# initially, all elements are in the same equivalence class, and no
		# attribute is computed
		classes_to_process = [elts]
		attributes_used = set()
		# the variable below will accumulate classes that will not be split further
		# because they are evaluated as final by function 'class_is_final'
		final_classes = []
		for attribute_name, attribute_f, split_f in splitters:
				#print('classes to process: {0}'.format(classes_to_process))
				classes_annotated = [(c, class_is_final(attributes_used, c))
														 for c in classes_to_process]
				classes_to_split = [x[0] for x in classes_annotated if not x[1]] #not final classes
				final_classes += [x[0] for x in classes_annotated if x[1]]
				#print('classes to split: {0}'.format(classes_to_split))
				#print('final classes: {0}'.format(final_classes))
				#print('next attribute: {}'.format(attribute_name))

				def f_class_split(class_file_list):
						elements_with_key = [(e, attribute_f(e)) for e in class_file_list]
						return split_f(elements_with_key)
				classes_to_process = [
						c for unsplit_class in classes_to_split for c in f_class_split(unsplit_class)]
				attributes_used.add(attribute_name)
				# print(final_classes)
		result = final_classes + classes_to_process
		#print('grouping final result: {}'.format(result))
		return result
