git config url."https://ouath2:${GIT_ACCESS_TOKEN}@git.cafebazaar.ir".insteadOf "https://git.cafebazaar.ir"


dvc repro dvc_dags/$QUEUE_NAME/dvc.yaml
dvc push
dvc metrics diff --show-md >> README.md

BRANCH_NAME="experiment_${QUEUE_NAME}_$(date "+%Y%m%d-%H%M%S")"
git checkout -b $BRANCH_NAME
git add .
git commit -m "dvc repro"

git push origin $BRANCH_NAME
