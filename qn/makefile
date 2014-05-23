all: quasinewton.pyx quasinewton.pxd setup.py
	python setup.py build_ext --inplace
	cp quasinewton.so ..

clean :
	@ ${RM} -rf quasinewton.so *.o quasinewton.c tokyo.so *.o tokyo.c