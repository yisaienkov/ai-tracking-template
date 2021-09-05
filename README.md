# ai-tracking-template


```bash
$ dvc run -n prepare \
    -p prepare \
    -d src/prepare.py -d resources/iris.csv \
    -o resources/out_prepare \
    python src/prepare.py
```

```bash
$ dvc run -n train \
    -p train \
    -d src/train.py -d resources/out_prepare \
    -o resources/out_train \
    python src/train.py
```

```bash
$ dvc run -n evaluate \
    -d src/evaluate.py -d resources/out_train -d resources/out_prepare \
    -M resources/out_evaluate/scores.json \
    python src/evaluate.py
```

```bash
$ dvc run -n prepare -p prepare -d src/prepare.py -d resources/iris.csv -o resources/out_prepare python src/prepare.py

$ dvc run -n train -p train -d src/train.py -d resources/out_prepare -o resources/out_train python src/train.py

$ dvc run -n evaluate -d src/evaluate.py -d resources/out_train -d resources/out_prepare -M resources/out_evaluate/scores.json python src/evaluate.py
```