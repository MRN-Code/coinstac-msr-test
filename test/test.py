# local_top.py

input:

python local.py '{"input":{"covariates":[[2,3],[3,4],[7,8],[7,5],[9,8]], "dependents":[6,7,8,5,6], "lambda":0}, "cache":{}}'


python local.py '{"input":{"covariates":[20.1,7.1,16.1,14.9,16.7,8.8,9.7,10.3,22,16.2,12.1,10.3], "dependents":[31.5,18.9,35,31.6,22.6,26.2,14.1,24.7,44.8,23.2,31.4,17.7], "lambda":0}, "cache":{}}'

X = [20.1,7.1,16.1,14.9,16.7,8.8,9.7,10.3,22,16.2,12.1,10.3]
y = [31.5,18.9,35,31.6,22.6,26.2,14.1,24.7,44.8,23.2,31.4,17.7]
X = sm.add_constant(X)
model = sm.OLS(y, X.astype(float)).fit()
model1 = sm.OLS(y, X.astype(float)).fit_regularized(L1_wt=0)

output:

{"input":{"covariates":[[2,3],[3,4],[7,8],[7,5],[9,8]], "dependents":[6,7,8,5,6], "lambda":0}, "cache":{}, "lastStepResult":{}}
{
    "cache": {
        "covariates": [
            [
                2,
                3
            ],
            [
                3,
                4
            ],
            [
                7,
                8
            ],
            [
                7,
                5
            ],
            [
                9,
                8
            ]
        ],
        "dependents": [
            6,
            7,
            8,
            5,
            6
        ],
        "lambda": 0
    },
    "output": {
        "beta_vector_local": [
            4.8169364881693655,
            -0.7310087173100865,
            1.0136986301369857
        ],
        "computation_phase": "local_1",
        "count_local": 5,
        "mean_y_local": 6.4,
        "ps_local": [
            0.017409543399922178,
            0.053517536133319554,
            0.04671291787951781
        ],
        "r_2_local": 0.9094740875562791,
        "ts_local": [
            7.479582268761776,
            -4.147193188256453,
            4.463105664143762
        ]
    }
}

# remote_top.py


