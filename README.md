
# Machine Learning  CS-433
## Class Project 1 | ML Higgs

Welcome to the repository!

This repository contains the code for  [Project 1](https://github.com/sansive/EPFL-CS433-Project1)  of the course of Machine Learning at EPFL.

The aim of this project has been to sort of recreating the process of “discovering” the Higgs particle, through the analysis of decay signature of protons collisions.

Using a huge amount of data and implementing machine learning methods seen in the course, we have been able to predict whether a collision was **signal** (a Higgs boson) or **background** (something else).

No extra libraries are needed to run the code but Numpy library, which can be installed with:
```
pip3 install --user numpy
```

This project has been developed in Python 3.9.
The running of the code can be done with:
```
python3 run.py
```
After running the code, the user will get as output a ```submission.csv ``` file, which will be located in the **data folder**.

## Structure

```
├── data
│   ├── submission.csv
│   ├── test.csv.zip
│   └── train.csv.zip
├── Description.pdf
├── README.md
├── report.pdf
└── scripts
    ├── implementations.py
    ├── proj1_helpers.py
    ├── project1.ipynb
    └── run.py
```

## Credits
This project has been developed by the team **Machakahomers** , composed by 3 ML students:
- Sandra Sierra ([@sansive](https://github.com/sansive))
- Belén Gómez ([@belengg27](https://github.com/belengg27))
- Rafael Mozo ([@rafaelmozo1](https://github.com/rafaelmozo1))

