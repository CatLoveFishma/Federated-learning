import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_epochs",default=5,help="本地训练的轮数")
    parser.add_argument("--r",default=0.2,help="本轮训练的客户的比例")
    parser.add_argument("--precision",default=5,help="训练过程中的数据的精度")
    parser.add_argument("--aggregate_epochs",default=10,help="聚合的轮数")
    args = parser.parse_args()
    return args