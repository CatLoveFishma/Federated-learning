import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_ep",default=10,help="本地训练的轮数")
    parser.add_argument("--r",default=0.1,help="本轮训练的客户的比例")
    parser.add_argument("--precision",default=5,help="训练过程中的数据的精度")
    parser.add_argument("--aggregate_epochs",default=10,help="聚合的轮数")
    parser.add_argument("--lr",default=0.001,help="学习率")
    parser.add_argument("--momentum",default=0.5,help="SGD momentum (default: 0.5)")
    parser.add_argument("--device",default='cpu',help="使用的设备")
    parser.add_argument("--local_bs",default=64,help="本地训练的batchsize")
    parser.add_argument("--betas",default=(0.99,0.999),help="need betas")
    parser.add_argument("--wd",default=1e-5,help="SGD momentum (default: 0.5)")
    parser.add_argument("--epochs",default=20,help="SGD momentum (default: 0.5)")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    args = parser.parse_args()
    return args



