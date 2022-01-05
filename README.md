# Portfolio

## 1. Pair Trading Strategy
### How to run
```
python PairTrading.py <stock1> <stock2> <start time> <end time> <transaction cost>
```

The `PairTrading.py` will download data of stock1 and stock2 ranging from the start time to the end time from yfinanace.

### Example
```
python PairTrading.py "2330.TW" "2303.TW" "2017-01-01" "2021-12-31" 0.00471
```

### Functionality
The program would plot three figures, which show the spread, cumulative return rate, and DD curve.

Besides, the time to open and clear position are labelled on the spread figure.

<br/>

## 2. SAT problem -- Clique Implementation
### How to run
Use `cd SAT_Clique` to get in the directory.

Use `make` to compile.

Use `./Clique input/<input file> <size of clique>` to run.

### Example
```
cd SAT_Clique

make

./Clique input/complete20V.in 8
```

### Functionality

The `Clique.cpp` will construct a graph according to the vertices and edges specified in the input file, and transform the graph problem into SAT problem.

Then it will use miniSAT to figure out whether there exist a clique with the size k in the graph.

<br/>

## 3. Quantum Communication -- Shor's Algorithm
### How to run
Use `conda create -n IBMQ python=3 jupyter` to set the IBMQ environment.

Use `conda activate IBMQ` to activate the environment.

Use `pip install qiskit matplotlib` to install the needed packages.

Use `jupyter notebook` to open the 'Shor_Alg.ipynb`


### Functionality

The `Shor_Alg` will perform Shor's algorithm to factorize a given number into two prime numbers. It is default to factorize 15.

Due to the qubits limitation, the maximum number to be factorize by the program is 65.
