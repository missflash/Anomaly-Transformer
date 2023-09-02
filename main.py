import os
import argparse

from torch.backends import cudnn
from utils.utils import *

from solver import Solver

import mlflow
from mlflow import log_metric, log_param, log_artifacts
from mlflow.exceptions import MlflowException


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    try:
        experiment = mlflow.get_experiment_by_name('anomaly_transformer')
        experiment_id = experiment.experiment_id
        print(experiment_id)
    except AttributeError:
        experiment_id = mlflow.create_experiment('anomaly_transformer', artifact_location='s3://mf/mlflow/')

    # with mlflow.start_run(experiment_id=experiment_id) as run:
    with mlflow.start_run(run_id=config.run_name) as run:
        mlflow.set_tracking_uri('http://mlflow-tracking.cloud.com')
        # mlflow.set_tag("mlflow.runName", now)
        # mlflow.autolog()
        for k, v in vars(config).items():
            log_param(k, v) if k != "mode" else print("Don't add mode!")

        cudnn.benchmark = True
        if (not os.path.exists(config.model_save_path)):
            mkdir(config.model_save_path)
        solver = Solver(vars(config))
    
        if config.mode == 'train':
            solver.train(mlflow)
        elif config.mode == 'test':
            solver.test(mlflow)
    
        return solver


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=4.00)
    parser.add_argument('--run_name', type=str, default=None)

    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
