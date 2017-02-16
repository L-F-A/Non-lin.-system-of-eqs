import numpy as np
import warnings

def Broyden(func,x0,B0,tol,tol_rel,Nmax,args,WarnM=False):
#########################################################################
#	Solving systems of nonlinear equations using Broyden method	#
#									#
#	Written by Louis-Francois Arsenault, Columbia University	# 
#			la2518@columbia.edu (2013-2017)			#
#									#
#		It follows the algorithm as given in			#
#	http://www.cs.illinois.edu/~heath/scicomp/notes/chap05.pdf 	#
#	from Prof. Michael T. Heath Department of Computer Science 	#
#		University of Illinois at Urbana-Champaign		#
#########################################################################
#									#
#Inputs:								#
#									#
# func     : the function to be used					#
# x0       : vector of initial solutions				#
# B0   	   : initial approximation of the Jacobian			#
# tol  	   : absolute tolerance						#
# tol_rel  : relative tolerance						#
# Nmax 	   : maximum number of iterations				#
# args     : tuple with all the extra parameters to be passed to func 	#
#	     in addition to x						#
# WarnM    : On screen warning message of no convergence after epochMax #
#	     iterations							#
#									#	
#Outputs:								#
#									#
# x 	   : the solution vector					#
# B 	   : the Jacobian at the solution				#
# Nite	   : number of iterations for convergence			#
#########################################################################
	
	x00=x0.copy()
	B00=B0.copy()

	N=1
	boucle=0

	if (type(args) is float) or (args is None):
		f0=func(x00)
	else:
		f0=func(x00,*args)

	while boucle==0:	
		s=np.linalg.solve(B00,-f0)
		x=x00+s
		if (type(args) is float) or (args is None):
			f1=func(x)
		else:
			f1=func(x,*args)
		y=f1-f0
		B00+=np.outer(y-B00.dot(s),s)/s.dot(s)
		test=np.abs(x-x00)
		testrel=test/(np.abs(x)+np.finfo(float).eps)

		if (np.max(testrel) < tol_rel) or (np.max(test)<tol):
			B=B00
			boucle=1
			Nite=N
		x00=x
		f0=f1
		if N==Nmax:
			if WarnM is True:
				warnings.warn('Max number of iterations; no convergence')
			boucle=1
			B=B00
			Nite=N
		N+=1

	return x,B,Nite
