# ####################################################################
#
#			  	   C Makefile
#	   Parallels and Distributed Systems
#				     Exercise 2
#
# Author: Triaridis Konstantinos <triaridis@ece.auth.gr>

# Adapted from:
# ......
#
# ####################################################################
#
#
# 'make lib'	build the libraries .a

# define the shell to bash
SHELL := /bin/bash

lib: gcc -c src/knnring_sequential.c -I inc -o sequential.o
		 mpicc -c src/knnring_mpi.c -I inc -o mpi.o
		 mpicc -c src/knnring_mpi_async.c -I inc -o mpi_async.o
		 ar rcs lib/knnring_sequential.a sequential.o
		 ar rcs lib/knnring_mpi.a mpi.o
		 ar rcs lib/knnring_mpi_async.a mpi_async.o
