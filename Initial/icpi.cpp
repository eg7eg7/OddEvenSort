#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <cstdlib>

#define SIZE 20

struct Particle
{
	float x;
	float y;
	int color;
};

void randStruct(struct Particle *mat, int size)
{
	for (int i = 0; i < size; i++)
	{
		mat[i].color = i;
		mat[i].x = (float)(rand() % 10000) / 31;
		mat[i].y = (float)(rand() % 10000) / 31;
	}
}

int compareAsc(struct Particle p1, struct Particle p2)
{
	if (p1.color > p2.color)
		return -1;
	else if (p1.color < p2.color)
		return 1;

	double p1_length = sqrt(p1.x*p1.x + p1.y * p1.y);
	double p2_length = sqrt(p2.x*p2.x + p2.y * p2.y);

	if (p1_length > p2_length)
		return -1;
	else if (p1_length < p2_length)
		return 1;
	return 0;
}

int compareDesc(struct Particle p1, struct Particle p2)
{
	return (-1)*compareAsc(p1, p2);
}

void printParticles(struct Particle * particles, int size) {
	for (int i = 0; i < size; i++)
	{
		printf("num %d\tcolor = %d\t\tx = %.2f\t\ty = %.2f\n", i, particles[i].color, particles[i].x, particles[i].y);
	}
}

//if no neighbour process, send -1
void oddEvenSort(int *location, int left, int right, MPI_Datatype ParticleMPIType, struct Particle *myParticle, int(*compar1)(struct Particle, struct Particle), struct Particle *otherParticle)
{
	int loc = *location;
	MPI_Status status;

	int(*compare)(struct Particle, struct Particle);
	compare = compar1;

	for (int i = 0; i < SIZE; i++)
	{
		if (i % 2 == 0)
		{
			if (loc % 2 == 0 && loc < SIZE - 1 && right > -1)
			{
				//receive from odd
				MPI_Sendrecv(myParticle, 1, ParticleMPIType, right, 0, otherParticle, 1, ParticleMPIType, right, 0, MPI_COMM_WORLD, &status);
				if (compare(*myParticle, *otherParticle) == -1)
				{
					*myParticle = *otherParticle;
				}
			}
			else if (loc % 2 == 1 && loc > 0 && left > -1)
			{
				//receive from even
				MPI_Sendrecv(myParticle, 1, ParticleMPIType, left, 0, otherParticle, 1, ParticleMPIType, left, 0, MPI_COMM_WORLD, &status);
				if (compare(*myParticle, *otherParticle) == 1)
				{
					(*myParticle) = (*otherParticle);
				}
			}
		}
		else
		{
			if (loc % 2 == 1 && loc > 0 && right > -1)
			{
				//receive from even
				MPI_Sendrecv(myParticle, 1, ParticleMPIType, right, 0, otherParticle, 1, ParticleMPIType, right, 0, MPI_COMM_WORLD, &status);
				if (compare(*myParticle, *otherParticle) == -1)
				{
					*myParticle = *otherParticle;
				}
			}
			else if (loc % 2 == 0 && loc < SIZE - 1 && left > -1)
			{
				//receive from odd
				MPI_Sendrecv(myParticle, 1, ParticleMPIType, left, 0, otherParticle, 1, ParticleMPIType, left, 0, MPI_COMM_WORLD, &status);
				if (compare(*myParticle, *otherParticle) == 1)
				{
					*myParticle = *otherParticle;
				}
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}
}

int main(int argc, char *argv[])
{
	struct Particle particle;
	int myrank, size;
	//MPI_Status status;
	MPI_Datatype ParticleMPIType;
	MPI_Datatype type[3] = { MPI_FLOAT, MPI_FLOAT, MPI_INT };
	int blocklen[3] = { 1, 1, 1 };
	MPI_Aint disp[3];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size != SIZE) {
		printf("Please run with %d processes.\n", SIZE); fflush(stdout);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	// Create MPI user data type for partical
	disp[0] = (char *)&particle.x - (char *)&particle;
	disp[1] = (char *)&particle.y - (char *)&particle;
	disp[2] = (char *)&particle.color - (char *)&particle;
	MPI_Type_create_struct(3, blocklen, disp, type, &ParticleMPIType);
	MPI_Type_commit(&ParticleMPIType);

	struct Particle particles[SIZE];
	if (myrank == 0)
	{
		randStruct(particles, SIZE);
	}

	struct Particle myParticle;
	struct Particle otherParticle;

	if (myrank == 0)
	{
		printf("before:\n");
		printParticles(particles, SIZE);
	}
	MPI_Scatter(particles, 1, ParticleMPIType, &myParticle, 1, ParticleMPIType, 0, MPI_COMM_WORLD);

	if (myrank > 0 && myrank < SIZE - 1)
		oddEvenSort(&myrank, myrank - 1, myrank + 1, ParticleMPIType, &myParticle, &compareDesc, &otherParticle);
	else if (myrank == 0 && myrank < SIZE - 1)
		oddEvenSort(&myrank, -1, myrank + 1, ParticleMPIType, &myParticle, &compareDesc, &otherParticle);
	else if (myrank > 0 && myrank == SIZE - 1)
		oddEvenSort(&myrank, myrank - 1, -1, ParticleMPIType, &myParticle, &compareDesc, &otherParticle);

	MPI_Gather(&myParticle, 1, ParticleMPIType, particles, 1, ParticleMPIType, 0, MPI_COMM_WORLD);
	if (myrank == 0)
	{
		printf("after:\n");
		printParticles(particles, SIZE);
	}

	MPI_Finalize();
	return 0;
}