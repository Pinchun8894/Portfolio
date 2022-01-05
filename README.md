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


## 2. SAT problem -- Clique Implementation
### How to run
Use `cd SAT_Clique` to get in the directory.

Use `make` to compile.

Use `./Clique input/<input file> <size of clique>` to run.