Clique: clean File.o Proof.o Solver.o Clique.o
	g++ -o $@ -g File.o Proof.o Solver.o Clique.o

File.o: sat/test/File.cpp
	g++ -c -g sat/test/File.cpp

Proof.o: sat/test/Proof.cpp
	g++ -c -g sat/test/Proof.cpp

Solver.o: sat/test/Solver.cpp
	g++ -c -g sat/test/Solver.cpp

satTest2.o: sat/test/satTest2.cpp
	g++ -c -g sat/test/satTest2.cpp

# clean:
# 	rm -f *.o satTest tags
clean:
	rm -f *.o Clique tags
