FC     = gfortran
FFLAGS = -O3 -fPIC -g -shared

OBJ    = fpbspl.o fpbisp.o bispeu.o
LIBOBJ = dfitpack_wrappers.o

dfitpack_wrappers.so: $(OBJ) $(LIBOBJ)
	$(FC) $(FFLAGS) $(OBJ) $(LIBOBJ) -o $@

%.o: %.f
	$(FC) $(FFLAGS) -c $*.f

dfitpack_wrappers.o: $(OBJ)
	$(FC) $(FFLAGS) -c dfitpack_wrappers.f90

.PHONY: all
all: dfitpack_wrappers.so

.PHONY: clean
clean:
	rm -rf *.o *.mod *.so
