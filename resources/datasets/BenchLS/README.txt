1) Introduction

	This package contains BenchLS, the Lexical Simplification benchmarking dataset described in the paper "Benchmarking Lexical Simplification Systems".

2) Content

	- README.txt: This file:
	- BenchLS.txt: The BenchLS benchmarking dataset.
	
3) Format

	Each line in the BenchLS.txt file follows the following format:
	
	<sentence> <complex_word> <position> <rank_1>:<candidate_substitution_1> ... <rank_n>:<candidate_substitution_n>
	
	Each component is separated by a tabulation marker.
	The <position> component refers to the token position of <complex_word> in <sentence>.
	The <rank_i> components refer to the simplicity ranking of <candidate_substitution_i>.
	The lower the ranking of <candidate_substitution_i>, the simpler it is.